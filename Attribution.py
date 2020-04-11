"""
(Classification &) Attribution component
"""

import uuid
from typing import Union, Tuple

import matplotlib.colors
import matplotlib.pylab as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import softmax
from tensorflow.python.keras import backend as K

from DataPreparation import ParserFrontend
from Helpers import *
from Training import predictions_to_label_numbers, GridSearchCheckpointInfo


def compute_gradients_of_output_wrt_to_input(model: tfkeras.Model, input_as_array: np.ndarray,
                                             output_neuron_index: int = None, handle_embeddings=True) \
        -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    # print(input_as_array.shape)
    sub_output_tensors = []
    with tensorflow.GradientTape() as tape:
        # Create tensor for input:
        input_as_tensor = tensorflow.constant(input_as_array)
        # Watch the input tensors to allow computation of gradients wrt to them (other tensors are watched
        # automatically):
        tape.watch(input_as_tensor)

        if handle_embeddings:
            # Create list for embedding results which will be filled during model call:
            model.embedding_results = []
        # Forward pass:
        output_tensor = model(input_as_tensor)
        assert len(output_tensor.shape) == 2 and output_tensor.shape[0] == 1, output_tensor.shape
        # Split output_tensor in tensors for each output node. This must be done inside the GradientTape context!
        for i in range(output_tensor.shape[1]):
            sub_output_tensor = output_tensor[:, i]
            sub_output_tensors.append(sub_output_tensor)
    # print("sp", sub_output_tensors)
    if output_neuron_index is None:
        # Differentiate complete output
        grad_target = output_tensor
    else:
        # Differentiate only one sub output
        grad_target = sub_output_tensors[output_neuron_index]
    # Source of the gradient:
    grad_sources = [input_as_tensor]
    if handle_embeddings:
        grad_sources += model.embedding_results
    # tape.gradient sums the gradients of all tensors in target! gradient(output_tensor) is approximately the same than
    # gradient(sub_output_tensors[0]) + gradient(sub_output_tensors[1])
    gradient_tensors = tape.gradient(target=grad_target, sources=grad_sources)
    # Convert each gradients to array:
    gradients = [grad_tensor.numpy() if grad_tensor is not None else None for grad_tensor in gradient_tensors]

    # Embeddings handling:
    if handle_embeddings:
        embeddings_inputs_and_gradients = []
        if len(gradients) > 1:
            # gradients contains input's gradients and embedding's gradients. Detect them:
            input_gradient = None
            embedding_gradients = []
            for gradient_array in gradients:
                assert gradient_array is not None
                if gradient_array.shape == input_as_array.shape:
                    assert input_gradient is None, input_gradient
                    input_gradient = gradient_array
                else:
                    # Rely on unchanged order for now. We will check whether the shapes match below
                    embedding_gradients.append(gradient_array)
            assert len(model.embedding_results) == len(embedding_gradients)

            for embedding_result_tensor, embedding_gradient in zip(model.embedding_results, embedding_gradients):
                # Make sure that the order of results matches the order of gradients (i.e. that tape.gradients
                # returns gradients such that the i'th gradient correspond to i'th source)
                assert embedding_result_tensor.shape == embedding_gradient.shape
                embeddings_inputs_and_gradients.append((embedding_result_tensor.numpy(), embedding_gradient))
        else:
            input_gradient = gradients[0]
        # Free embeddings' results:
        model.embedding_results = None
        # Check and return gradients:
        assert input_gradient is not None and input_gradient.shape == input_as_array.shape
        return input_gradient, embeddings_inputs_and_gradients
    else:
        assert len(gradients) == 1, gradients
        return gradients[0], []


def compute_integrated_gradients(model, input: Union[np.ndarray, List[np.ndarray]], output_neuron_index, steps,
                                 baseline: Union[int, np.ndarray] = 0,
                                 embedding_baselines_arg: Union[int, List[np.ndarray]] = 0,
                                 handle_embeddings=True,
                                 has_multiple_inputs=False):
    """
    from: https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    :param model:
    :param input:
    :param output_neuron_index:
    :param steps:
    :param baseline:
    :param embedding_baselines_arg:
    :param handle_embeddings:
    :param has_multiple_inputs:
    :return:
    """
    assert not isinstance(input, list), "list not supported yet"
    if not isinstance(baseline, np.ndarray):
        # Make baseline of input's shape
        baseline = to_numpy_array(np.full(input.shape, baseline))

    assert input.shape == baseline.shape

    summed_input_gradient = None
    summed_embeddings_gradients = []
    embedding_inputs = []
    embedding_baselines = []
    print("Computing integrated gradients for output neuron ", output_neuron_index, ": ", sep="", end="")
    for alpha in list(np.linspace(1. / steps, 1.0, steps)):
        input_mod = baseline + (input - baseline) * alpha
        input_gradient, embeddings_inputs_and_gradients = compute_gradients_of_output_wrt_to_input(model,
                                                                                                   input_mod,
                                                                                                   output_neuron_index,
                                                                                                   handle_embeddings)
        # Input:
        if summed_input_gradient is None:
            summed_input_gradient = input_gradient
        else:
            summed_input_gradient = summed_input_gradient + input_gradient
        # Embeddings:
        for embed_index, embeddings_input_and_gradient in enumerate(embeddings_inputs_and_gradients):
            if len(summed_embeddings_gradients) <= embed_index:
                assert len(summed_embeddings_gradients) == embed_index
                summed_embeddings_gradients.append(embeddings_input_and_gradient[1])
            else:
                summed_embeddings_gradients[embed_index] += embeddings_input_and_gradient[1]

            # print(input_mod[:,embed_index], np.mean(embeddings_input_and_gradient[0]))
            # Embedding input and baseline when alpha == 1, i.e., normal input
            if alpha == 1:
                assert (input_mod == input).all()
                embedding_inputs.append(embeddings_input_and_gradient[0])
                if isinstance(embedding_baselines_arg, int):
                    embedding_baselines.append(to_numpy_array(np.full(embeddings_input_and_gradient[0].shape,
                                                                      embedding_baselines_arg)))
                else:
                    embedding_baselines.append(embedding_baselines_arg[embed_index])
        print("#", end="")
    assert len(embedding_inputs) == len(summed_embeddings_gradients)
    assert len(embedding_baselines) == len(summed_embeddings_gradients)

    print("")
    input_integrated_gradient = summed_input_gradient * (input - baseline) / steps
    embedding_input_and_integrated_gradients \
        = [(embedding_input, summed_embeddings_gradient * (embedding_input - embedding_baseline) / steps)
           for embedding_input, embedding_baseline, summed_embeddings_gradient
           in zip(embedding_inputs, embedding_baselines, summed_embeddings_gradients)]

    return input_integrated_gradient, embedding_input_and_integrated_gradients


def make_tikz_attribution_rectangle(start: Union[Tuple[float, float], List], end: Union[Tuple[float, float], List],
                                    opacity: float, colorstyle: str, text: str, origin="CodeOrigin"):
    assert 0 <= opacity <= 1, opacity
    return ("\\draw [attributionrect,opacity=\\maxopacity*"
            + str(opacity) + "," + colorstyle
            + "] ($ (" + origin + ") + " + str(tuple(start))
            + " $) rectangle ($ (" + origin + ") + " + str(tuple(end))
            + " $) node[attributiontext] {" + get_latex_escaped(text) + "};")


def make_tikz_code_visualization(ast_nodes: List[ParserFrontend.AstNodeType], importance_list: List[float],
                                 texts: List[str] = None, predicted_class_is_non_bug_class: bool = None,
                                 caption: str = ""):
    if texts is None:
        texts = [""] * len(ast_nodes)
    assert predicted_class_is_non_bug_class is not None
    pos_or_neg_colorstyle = "positivecolor" if not predicted_class_is_non_bug_class else "negativecolor"

    attribution_infos = {}
    max_abs_importance = 0
    for ast_node, importance, text in zip(ast_nodes, importance_list, texts):
        main_file_path = None
        if hasattr(ast_node, "tu"):
            main_file_path = ast_node.tu.id.spelling
            if isinstance(main_file_path, tuple):
                assert len(main_file_path) == 3, main_file_path
                # AST nodes were created from in-memory source code snippet. In this case main_file_path is a tuple
                # of the temporary file, the source code snippet, and its extension. Use the source code with
                # extension instead of path:
                main_file_path = main_file_path[1] + "\n" + main_file_path[2]

        if hasattr(ast_node, "extent"):
            loc_start = ast_node.extent.start
            loc_end = ast_node.extent.end
            loc_file_path = loc_start.file
            is_main = False
            if loc_file_path == ":MN:":
                assert main_file_path is not None
                loc_file_path = main_file_path
                is_main = True
            attribution_info = attribution_infos.setdefault(loc_file_path,
                                                            {"rectangles": [], "importances": [], "texts": []})
            attribution_info["rectangles"].append(((loc_start.col, loc_start.line), (loc_end.col, loc_end.line)))
            attribution_info["importances"].append(importance)
            max_abs_importance = max(max_abs_importance, abs(importance))
            attribution_info["texts"].append(text)
        else:
            assert ast_node.kind[0].startswith("EndOf"), ast_node

    def importance_to_opacity(importance):
        return (importance / max_abs_importance) ** 2

    source_code = ""
    tiks_commands = []
    file_line_offset = 0
    for file_path, attribution_info in attribution_infos.items():
        if "\n" in file_path:
            file_source_code = "\n".join(file_path.split("\n")[:-1])
        else:
            file_source_code = read_file_content(file_path)
        file_source_code_lines = file_source_code.split("\n")
        for rectangle, importance, text in zip(attribution_info["rectangles"], attribution_info["importances"],
                                               attribution_info["texts"]):

            start = list(rectangle[0])
            end = list(rectangle[1])

            # lines and columns are 1 based. Make them zero based:
            start[0] -= 1
            end[0] -= 1
            start[1] -= 1
            end[1] -= 1

            if start[1] != end[1]:
                # Multi line ast node (e.g. if structure):
                # Determine smallest col number with source code char and largest col number with source code char:
                min_start_in_between = start[0]
                max_end_in_between = 0
                for in_between_line in range(start[1], end[1]):
                    start_of_line = 0
                    for line_char in file_source_code_lines[in_between_line]:
                        if not line_char.isspace():
                            break
                        start_of_line += 1
                    min_start_in_between = min(min_start_in_between, start_of_line)
                    max_end_in_between = max(max_end_in_between, len(file_source_code_lines[in_between_line].rstrip()))
                end[0] = max_end_in_between - 1
                start[0] = min_start_in_between
            else:
                # Extend ranges that end on the beginning of an token (e.g. 2 for 219 or ' for 'e')
                last_char = file_source_code_lines[start[1]][end[0]]
                loc_len = end[0] - start[0] + 1

                # print("prev|", file_source_code_lines[start[1]][start[0]:end[0] + 1], "|", text)

                def is_cont_char(char):
                    return char.isalnum() or char in ("_", "'", "\"")

                if is_cont_char(last_char):
                    in_string = False
                    if loc_len == 1 and last_char in ("'", "\""):
                        in_string = True
                    new_end = end[0] + 1
                    while True:
                        next_char = file_source_code_lines[start[1]][new_end]
                        if in_string:
                            if next_char in ("'", "\""):
                                # Add the ending quote and stop:
                                new_end += 1
                                break
                            elif next_char == "\\":
                                # Skip the escaped char
                                new_end += 1
                            else:
                                # Continue with next char
                                pass
                        elif not is_cont_char(next_char):
                            break
                        new_end += 1
                        continue

                    end[0] = new_end - 1
                    # print("new|", file_source_code_lines[start[1]][start[0]:end[0] + 1], "|")

            # Add line offset in case of multiple source files
            start[1] += file_line_offset
            end[1] += file_line_offset
            # intervals are inclusive:
            end[0] += 1
            # rectangle height:
            end[1] += 1
            # line_height = line_heights.setdefault(start[1], [1.0])
            # line_height[0] -= 0.1
            # if line_height[0] <= 0:
            #     line_height[0] = 1.05
            # end[1] += line_height[0]

            # Invert y values (y is going up in tikz)
            start[1] *= -1
            end[1] *= -1

            # TODO: Do not show the text for now:
            text = ""

            tiks_commands.append(make_tikz_attribution_rectangle(start, end, importance_to_opacity(importance),
                                                                 pos_or_neg_colorstyle if importance > 0 else "neutralcolor",
                                                                 text))

        source_code += "\n".join(file_source_code_lines)
        file_line_offset += len(file_source_code_lines) - 1  # -1 as "\n".join does not add \n for last line

    # Legend:
    tiks_commands.append(make_tikz_attribution_rectangle((0, 0), (8, - 1), 1.0, "neutralcolor", "", "LegendNegOrigin"))
    tiks_commands.append(make_tikz_attribution_rectangle((0, 0), (4, - 1), 0.0, "neutralcolor", "", "LegendZeroOrigin"))
    tiks_commands.append(make_tikz_attribution_rectangle((0, 0), (8, - 1), 1.0, pos_or_neg_colorstyle, "",
                                                         "LegendPosOrigin"))

    # legend_width = 40
    # start_y = -file_line_offset
    # curr_importance = -max_abs_importance
    # for curr_start_x in range(0, legend_width + 2, 2):
    #     curr_importance = -max_abs_importance + curr_start_x * 2 * max_abs_importance / legend_width
    #     tiks_commands.append(make_tikz_attribution_rectangle((curr_start_x, start_y), (curr_start_x + 2, start_y - 1),
    #                                                          importance_to_opacity(curr_importance),
    #                                                          pos_or_neg_colorstyle if curr_importance > 0 else "neutralcolor",
    #                                                          "{:.1f}".format(importance_to_opacity(curr_importance))))

    guid_str = uuid.uuid4().hex
    tikzmark_code_start_id = "start" + guid_str
    tikzmark_legend_neg_start_id = "legendneg" + guid_str
    tikzmark_legend_zero_start_id = "legendzero" + guid_str
    tikzmark_legend_pos_start_id = "legendnpos" + guid_str
    latex_code = ("\\begin{tikzpicture}[remember picture, attribution scale]\n"
                  + "\\coordinate [attribution shift] (CodeOrigin) at ($ (pic cs:" + tikzmark_code_start_id + ") $);\n"
                  + "\\coordinate [attribution shift] (LegendNegOrigin) at ($ (pic cs:" + tikzmark_legend_neg_start_id + ") $);\n"
                  + "\\coordinate [attribution shift] (LegendZeroOrigin) at ($ (pic cs:" + tikzmark_legend_zero_start_id + ") $);\n"
                  + "\\coordinate [attribution shift] (LegendPosOrigin) at ($ (pic cs:" + tikzmark_legend_pos_start_id + ") $);\n"
                  # + "\\node [overlay] at (CodeOrigin) {\\textbullet};\n"
                  # + "\\draw [overlay, help lines] (CodeOrigin) grid ++(20, -20);\n"
                  # + "\\draw [overlay, blue] ($ (CodeOrigin) + (0, 0) $) rectangle ($ (CodeOrigin) + (1, -1) $);\n"
                  # + "\\draw [overlay, green] ($ (CodeOrigin) + (4, -4) $) rectangle ($ (CodeOrigin) + (5, -3) $);\n"
                  # + "\\draw [overlay, pink] ($ (CodeOrigin) + (4, -8) $) rectangle ($ (CodeOrigin) + (5, -7) $);\n"
                  + ("\n".join(tiks_commands))
                  + "\n\\end{tikzpicture}\n"
                  + "\\begin{minted}[escapeinside=\\\\$\\\\$]{c++}\n"
                  + "$\\tikzmark{" + tikzmark_code_start_id + "}$"
                  + source_code + ("\n" if source_code[-1] != "\n" else "")  # Ensure newline before \end{minted}
                  + "\\end{minted}\n"
                  )
    return ("\\begin{listing}\n"
            + latex_code + "\n"
            # It is necessary to have a short caption and to have the tikz commands only in the long caption! Otherwise
            #  compile errors occur.
            + "\\caption[" + get_latex_escaped(caption) + "]{" + get_latex_escaped(caption)
            + "The overlay color corresponds to the influence on the prediction, "
            + "from \\tikzmark{" + tikzmark_legend_neg_start_id + "}\legendinline{negative} influence "
            + "over \\tikzmark{" + tikzmark_legend_zero_start_id + "}\legendinline{zero} influence "
            + "to \\tikzmark{" + tikzmark_legend_pos_start_id + "}\legendinline{positive} influence.} \n"
            + "\\label{list:label}\n"
            + "\\end{listing}")


def make_latex_document_code(content_code):
    return ("\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[paperheight=50in,paperwidth=8.5in]{geometry}\n"
            "\\usepackage{tikz}\n"
            "\\usetikzlibrary{calc}\n"
            "\\usetikzlibrary{math}\n"
            "\\usetikzlibrary{tikzmark}\n"
            "\\usepackage{minted}\n"
            "\\newmintinline[legendinline]{cpp}{}\n"
            "\n"
            "\\tikzset{\n"
            "  attribution scale/.style={xscale=0.185, yscale=0.4225},\n"  # In thesis: xscale=0.217, yscale=0.51
            "  % Shift is affected by scale!\n"
            "  attribution shift/.style={shift={(0,0.38)}},\n"  # In thesis: shift={(0,0.3825)}
            "  attributionrect/.style={overlay, rectangle, rounded corners, opacity=0.5, fill},\n"
            "  positivecolor/.style={red},\n"  # positive = bug found
            "  neutralcolor/.style={blue!70!white},\n"
            "  negativecolor/.style={green!70!black},\n"
            "  attributiontext/.style={font=\\tiny, pos=.5}\n"
            "}\n"
            "\\tikzmath{\\maxopacity=0.8;}\n"
            "\n"
            "\\begin{document}\n"
            + content_code + "\n"
            + "\\end{document}"
            )


def visualize_grads_of_output_wrt_to_input(gridsearch_checkpoint: GridSearchCheckpointInfo, feature_sequence_array,
                                           corresponding_ast_nodes, actual_label_number=None,
                                           use_absolute_values=False,
                                           # show_plot=True,
                                           feat_vec_components_to_show=None,
                                           save_plot_file_name_prefix=None,
                                           visualization_dir_path=None,
                                           add_embeddings=True,
                                           multiplicate_with_input=True,
                                           integrated_gradients_baseline=None,
                                           integrated_gradients_steps=100,
                                           threshold_to_use="argmax",
                                           apply_softmax=True,
                                           skip_gradient_calculation=False):
    """
    Performs classification with given model of given feature sequence and generates visualizations of corresponding
     attributions. Two PDFs are generated and stored in the model's directory. Firstly, a PDF with the feature sequence
     and its attributions in a tabular format is generated using matplotlib (or bokeh). Secondly, a PDF with attribution
     overlays placed over the associated source code is generated using Latex and TikZ.
    :param gridsearch_checkpoint:
    :param feature_sequence_array:
    :param corresponding_ast_nodes:
    :param actual_label_number:
    :param use_absolute_values:
    :param feat_vec_components_to_show:
    :param save_plot_file_name_prefix:
    :param visualization_dir_path:
    :param add_embeddings:
    :param multiplicate_with_input:
    :param integrated_gradients_baseline:
    :param integrated_gradients_steps:
    :param threshold_to_use:
    :param apply_softmax:
    :param skip_gradient_calculation:
    :return:
    """
    # Both matplotlib and bokeh have problems to generate consistent layouts for different feature sequences.
    #  matplotlib's pdf generation is more convenient than bokeh's.
    USE_BOKEH = False
    make_sub_plot_pdf_page = make_sub_plot_bokeh if USE_BOKEH else make_sub_plot_pdfpage_matplotlib

    assert len(feature_sequence_array.shape) == 2, "Expected one 2D feature sequence"
    if integrated_gradients_baseline is not None:
        assert multiplicate_with_input, "Integrated Gradients uses multiplication with input," \
                                        "but multiplicate_with_input was set to False"

    feat_vec_infos_to_use \
        = gridsearch_checkpoint.dataset_custodian.frontend.get_feat_vec_component_info(feat_vec_components_to_show)

    corresponding_ast_nodes_loc_summary_str = ParserFrontend.get_ast_nodes_location_summary_string(
        corresponding_ast_nodes)
    print("Corresponding file locations and line number intervals:")
    print(corresponding_ast_nodes_loc_summary_str)

    start_time = time.time()

    feature_sequence_array_as_one_element_list = to_numpy_array([feature_sequence_array])
    ### Get prediction (corresponds to K.get_session().run(model.layers[-1].output, feed_dict =
    # {model.inputs[0].name: test_sample_as_array})
    y_pred = gridsearch_checkpoint.model.predict(feature_sequence_array_as_one_element_list)
    if apply_softmax:
        y_pred = softmax(y_pred, axis=1)
    print("Using threshold", threshold_to_use, end="")
    predicted_label_numbers = predictions_to_label_numbers(y_pred, threshold_to_use)
    assert predicted_label_numbers.shape == (1,)
    predicted_label_number = predicted_label_numbers[0]
    predicted_label = gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[predicted_label_number]
    prediction_certainties = y_pred[0]
    if not apply_softmax:
        prediction_certainties = softmax(prediction_certainties)
    prediction_certainty_str = "{:.2f}%".format(prediction_certainties[predicted_label_number] * 100)
    print(" and predicted the label \"", predicted_label, "\" with ", prediction_certainty_str, " certainty (",
          prediction_certainties, ")", sep="")
    predicted_class_is_non_bug_class = predicted_label_number == len(
        gridsearch_checkpoint.dataset_custodian.int_to_label_mapping) - 1
    actual_label_str = None
    prediction_is_correct = None
    if actual_label_number is not None:
        actual_label = gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[actual_label_number]
        prediction_is_correct = actual_label == predicted_label
        actual_label_str = "Actual label: " + actual_label + " ==> Prediction is " \
                           + ("correct" if prediction_is_correct else "wrong")
        print(" ", actual_label_str)

    print("Time after prediction:", time.time() - start_time, "s")

    if skip_gradient_calculation:
        print("Skipping gradient calculation and visualization...")
        return [], []

    sub_visualizations = []

    if tensorflow.executing_eagerly():
        output_node_count = y_pred.shape[1]
        list_of_gradient_values_as_array = []
        for i in range(output_node_count):
            if integrated_gradients_baseline is not None:
                input_gradient, embeddings_inputs_and_gradients \
                    = compute_integrated_gradients(gridsearch_checkpoint.model,
                                                   feature_sequence_array_as_one_element_list,
                                                   output_neuron_index=i,
                                                   steps=integrated_gradients_steps,
                                                   baseline=integrated_gradients_baseline,
                                                   handle_embeddings=add_embeddings,
                                                   has_multiple_inputs=False)
            else:
                input_gradient, embeddings_inputs_and_gradients \
                    = compute_gradients_of_output_wrt_to_input(gridsearch_checkpoint.model,
                                                               feature_sequence_array_as_one_element_list, i,
                                                               handle_embeddings=add_embeddings)

            # Replace parts of input that contain the number that is used as input to embedding with aggregated value
            # of corresponding embeddings.
            sub_sub_visualizations = []
            for index, (embedding_result, embedding_gradient) in enumerate(embeddings_inputs_and_gradients):
                assert np.count_nonzero(input_gradient[:, :, index]) == 0
                # Use mean of embedding gradients as single gradient value for input gradients
                input_gradient[:, :, index] = np.mean(embedding_gradient, axis=2)

                # Add sub-visualization including a title for the heatmap:
                sub_sub_visualizations.append([embedding_result,
                                               embedding_gradient,
                                               "Heatmap for embedding \"" + feat_vec_infos_to_use[index].name + "\" of "
                                               + str(i) + "'th output node (for label \""
                                               + gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[i]
                                               + "\")"])

            sub_visualizations.append(sub_sub_visualizations)
            list_of_gradient_values_as_array.append(input_gradient)

    else:
        # Old non-eager implementation
        print("Computing gradients of " + str(gridsearch_checkpoint.model.layers[-1].output) + " with respect to "
              + str(gridsearch_checkpoint.model.layers[0].input) + " ...")
        # Gradient tensor of each output node with respect to input:

        gradient_tensors = []
        # Create gradient tensor for each output node:
        output_node_count = gridsearch_checkpoint.model.layers[-1].output.get_shape().as_list()[1]
        for i in range(output_node_count):
            # output[0] or output[0, :] returns the output for first (zero based) sample in batch. Therefore the number must be zero because there is only one sample. Otherwise there will be an error during execution
            gradient_array = K.gradients(gridsearch_checkpoint.model.layers[-1].output[0, i],
                                         gridsearch_checkpoint.model.layers[0].input)
            assert len(gradient_array) == 1, "Expected one gradient tensor only"
            gradient_tensors.append(gradient_array[0])

        # Evaluate all gradient tensors at once:
        feed_dict = {gridsearch_checkpoint.model.inputs[0].name: feature_sequence_array_as_one_element_list}
        list_of_gradient_values_as_array = K.get_session().run(gradient_tensors, feed_dict=feed_dict)
        # print(list_of_gradient_values_as_array[0])
        # Should be one gradient array only:
        assert len(list_of_gradient_values_as_array) == output_node_count, "Expected " + str(
            output_node_count) + " arrays of gradients"

    # Only keep requested parts of the gradient and input:
    slice_indices = []
    for feat_vec_info in feat_vec_infos_to_use:
        for slice_index in range(feat_vec_info.index_start, feat_vec_info.index_end):
            slice_indices.append(slice_index)
    # This also influences the tick labels:
    x_tick_positions = []
    x_tick_labels = []
    # Because the input and gradients may have been sliced, the absolute indices can not be used:
    curr_index = 0
    for feat_vec_info in feat_vec_infos_to_use:
        x_tick_positions.append(curr_index)
        x_tick_labels.append(feat_vec_info.name)
        # Advance index by length of current input feature
        curr_index += feat_vec_info.index_end - feat_vec_info.index_start

    # Remove undesired parts of the input sample:
    feature_sequence_array = feature_sequence_array[:, slice_indices]
    for i in range(output_node_count):
        # Remove first dimension (which corresponds to the batch size which is one here)
        assert list_of_gradient_values_as_array[i].shape[0] == 1, \
            "Expected batch size to be one for each array of gradients"
        list_of_gradient_values_as_array[i] = list_of_gradient_values_as_array[i][0]
        for sub_sub_visualizations in sub_visualizations[i]:
            assert sub_sub_visualizations[0].shape[0] == 1
            assert sub_sub_visualizations[1].shape[0] == 1
            sub_sub_visualizations[0] = sub_sub_visualizations[0][0]
            sub_sub_visualizations[1] = sub_sub_visualizations[1][0]

        # Remove undesired parts of the input sample:
        list_of_gradient_values_as_array[i] = list_of_gradient_values_as_array[i][:, slice_indices]

        # Print some statistics:
        non_zero_count = np.where(list_of_gradient_values_as_array[i] != 0)[0].shape[0]
        nan_count = np.where(np.isnan(list_of_gradient_values_as_array[i]))[0].shape[0]
        overall_count = list_of_gradient_values_as_array[i].size
        print("Count of {}th output neuron's gradient values that differ from zero: {}/{} ({:.2f}%)"
              .format(i, non_zero_count, overall_count, non_zero_count / overall_count * 100))
        if nan_count > 0:
            print("WARNING: Count of {}th output neuron's gradient values that are NaN: {}/{} ({:.2f}%)"
                  .format(i, nan_count, overall_count, nan_count / overall_count * 100))

        if multiplicate_with_input:
            # Multiply gradient with input ( a * b is elementwise multiplication if a and b are numpy arrays)
            list_of_gradient_values_as_array[i] *= feature_sequence_array
            for sub_sub_visualizations in sub_visualizations[i]:
                sub_sub_visualizations[1] *= sub_sub_visualizations[0]

        # Convert to absolute values if requested:
        if use_absolute_values:
            list_of_gradient_values_as_array[i] = np.abs(list_of_gradient_values_as_array[i])
            for sub_sub_visualizations in sub_visualizations[i]:
                sub_sub_visualizations[0] = np.abs(sub_sub_visualizations[0])
                sub_sub_visualizations[1] = np.abs(sub_sub_visualizations[1])

    print("Time after gradients:", time.time() - start_time, "s")
    # Visualize gradients as heatmap:
    if save_plot_file_name_prefix is not None:
        plt.rc('font', family='DejaVu Sans')  # use font that contains more unicode characters

        if visualization_dir_path is None:
            visualization_dir_path = os.path.join(gridsearch_checkpoint.saved_model_dir_path, "Visualizations")
        create_dir_if_necessary(visualization_dir_path)
        save_plot_file_name = as_safe_filename(save_plot_file_name_prefix + "_pred_" + predicted_label + "_"
                                               + prediction_certainty_str + "certain_"
                                               + str(prediction_certainties)[:100]
                                               + (("intgrad" + str(integrated_gradients_steps))
                                                  if integrated_gradients_baseline is not None else "")) \
                              + "_heatmap.pdf"
        save_plot_file_path = os.path.join(visualization_dir_path, save_plot_file_name)
        save_latex_file_path = os.path.join(visualization_dir_path, save_plot_file_name.replace("_heatmap.pdf",
                                                                                                "_overlay.tex"))

        with PdfPages(save_plot_file_path) as pdf_pages:
            # fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
            # max_sub_sub_length = max([len(sub_sub_vis) for sub_sub_vis in sub_visualizations])
            # rows_per_sub_vis = math.ceil(max_sub_sub_length / output_node_count)
            # # First one for title
            # gs = GridSpec(nrows=1 + len(sub_visualizations),
            #               ncols=max(output_node_count, max_sub_sub_length),
            #               figure=fig)
            # # fig, axes = plt.subplots(nrows=1, ncols=output_node_count, dpi=120)
            # plt.subplots_adjust(wspace=0.7)
            # fig.subplots_adjust(bottom=0)
            # fig.subplots_adjust(top=1)
            # fig.subplots_adjust(right=1)
            # fig.subplots_adjust(left=0)

            main_heading = ("Attribution heatmaps visualizing"
                            + ((" integrated (" + str(integrated_gradients_steps) + ")")
                               if integrated_gradients_baseline is not None
                               else (" input *" if multiplicate_with_input else ""))
                            + " gradients of model outputs w.r.t. green input\n\nThe model predicted the label \""
                            + predicted_label + "\" for the green input\nusing threshold " + str(threshold_to_use)
                            + " with " + prediction_certainty_str + " certainty\n(" + str(prediction_certainties)
                            + ")\n" + ((actual_label_str + "\n") if actual_label_str is not None else "")
                            + "\nCorresponding file locations and line number intervals for green input:\n"
                            + corresponding_ast_nodes_loc_summary_str)

            add_text_pdf_page(pdf_pages, main_heading)

            # Custom colormap with black in the middle
            # cmap_blueblackred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "blue", "black",
            #                                                                              "red", "lightcoral"])
            cmap_blackgreen_simple = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "green"])
            cmap_bluewhitered = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "lightblue", "white",
                                                                                         "lightcoral", "red"])

            listing_codes = []
            # cell_amount_per_axis = int(gs._ncols / output_node_count)
            for i in range(output_node_count):
                curr_output_node_label = gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[i]

                # # Current axis object:
                # axes_obj = fig.add_subplot(gs[0, i * cell_amount_per_axis: (i + 1) * cell_amount_per_axis])

                # Number of padded feat vecs:
                padded_count = feature_sequence_array.shape[0] - len(corresponding_ast_nodes)
                # Create annotation strings:
                annotation_strings = []
                # Currently each heatmap is on a separate page and needs all information:
                if True or i == output_node_count - 1:
                    # Use AST-Node-String-Representation as Tick-Labels
                    node_depths = [node.depth for node in corresponding_ast_nodes]
                    min_depth = min(node_depths)
                    # test_sample_as_array might be padded:
                    annotation_strings.extend(["― <padding>" for i in range(padded_count)])
                    for node, node_depth in zip(corresponding_ast_nodes, node_depths):
                        # print_tree(node)
                        node_as_string = \
                            gridsearch_checkpoint.dataset_custodian.frontend.get_node_as_string(node, compact=True)
                        # limit max len of tick label:
                        node_as_string = node_as_string[0:150]
                        annotation_strings.append("―" * (node_depth - min_depth + 1) + " " + node_as_string)
                else:
                    annotation_strings = ["―" for y_pos in range(feature_sequence_array.shape[0])]

                # Compute the importance rank of each feat vec / ast node, i.e. how high is the sum of gradients
                # compared to other nodes (np.argsort returns a list of indices. The i'th element in argsort's result
                # specifies the position where the i'th sum must be moved to to create a sorted list. Therefore, the
                # index/position of i inside the array can be interpreted as importance of the i'ths feat vec / ast
                # node. (Higher importance is more important.))
                node_importance_positions = list(np.argsort(np.sum(list_of_gradient_values_as_array[i],
                                                                   axis=1)))
                node_abs_importance_positions = list(np.argsort(np.sum(np.abs(list_of_gradient_values_as_array[i]),
                                                                       axis=1)))
                importances_count = len(node_importance_positions)
                assert importances_count == len(corresponding_ast_nodes)
                node_importances = [node_importance_positions.index(index) - importances_count / 2
                                    for index in range(len(node_importance_positions))]
                node_abs_importances = [node_abs_importance_positions.index(index)
                                        for index in range(len(node_abs_importance_positions))]

                # Number of digits for node number and importance rank
                y_label_number_digits = int(math.log10(len(annotation_strings))) + 1
                y_label_number_format_str = "{:0" + str(y_label_number_digits) + "d}"

                y_tick_positions = np.arange(0, len(annotation_strings), 1)
                y_tick_labels = [y_label_number_format_str.format(y_pos) + "("
                                 + y_label_number_format_str.format(
                    node_abs_importances[y_pos]) + ")" + annotation_string
                                 for y_pos, annotation_string in enumerate(annotation_strings)]

                # Specify how each tick label should look like:
                # Color the tick labels to represent it's correspondings ast node's importance for the classification
                # result:
                scalar_to_cmap = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=len(node_abs_importances) - 1),
                                                   cmap=cmap_blackgreen_simple)
                y_tick_attributes = []
                for node_importance in node_abs_importances:
                    attribute_dict = {"weight": "bold", "text_font_style": "bold",
                                      "color": scalar_to_cmap.to_rgba(node_importance),
                                      "text_color": scalar_to_cmap.to_rgba(node_importance)}
                    y_tick_attributes.append(attribute_dict)

                heading = "Heatmap for " + str(i) + "'th output node (for label \"" + curr_output_node_label + "\")"

                # Create overlay Latex and Tikz code:
                listing_codes.append(make_tikz_code_visualization(
                    corresponding_ast_nodes, node_importances, texts=annotation_strings,
                    predicted_class_is_non_bug_class=predicted_class_is_non_bug_class,
                    caption=heading.replace("Heatmap", "Prediction: " + predicted_label + "; Importance")
                            + ". "
                            + "Softmax output: " + str(prediction_certainties[i]) + ". "
                            + "Min: " + str(np.min(list_of_gradient_values_as_array[i])) + ". "
                            + "Max: " + str(np.max(list_of_gradient_values_as_array[i])) + ". "
                            + "Mean: " + str(np.mean(list_of_gradient_values_as_array[i])) + ". "
                            + "Std: " + str(np.std(list_of_gradient_values_as_array[i]))
                            + ". ")
                )

                # Create the heatmap with matplotlib (or bokeh):
                make_sub_plot_pdf_page(pdf_pages, feature_sequence_array, list_of_gradient_values_as_array[i],
                                       heading,
                                       True,
                                       x_tick_positions, x_tick_labels,
                                       y_tick_positions, y_tick_labels, y_tick_attributes,
                                       'hot' if use_absolute_values else cmap_bluewhitered)

            # Sub visualization heatmaps (e.g. Embeddings):
            for sub_vis_index, sub_visualization in enumerate(sub_visualizations):
                for sub_sub_vis_index, sub_sub_visualization in enumerate(sub_visualization):
                    # axes_obj = fig.add_subplot(gs[1 + (sub_vis_index * rows_per_sub_vis)
                    #                               + int(sub_sub_vis_index / output_node_count),
                    #                               sub_sub_vis_index % output_node_count])
                    # axes_obj = fig.add_subplot(gs[1 + sub_vis_index,
                    #                               sub_sub_vis_index])
                    make_sub_plot_pdf_page(pdf_pages, sub_sub_visualization[0], sub_sub_visualization[1],
                                           sub_sub_visualization[2],
                                           color_map='hot' if use_absolute_values else cmap_bluewhitered)

            # Create complete latex file from heading, listings, and TikZ commands:
            write_file_content(save_latex_file_path,
                               make_latex_document_code(
                                   get_latex_escaped(main_heading.replace("heatmap", "overlay").replace("green ", ""))
                                   + "\n\\clearpage" + "\n".join(listing_codes)))
            print("Saved latex to", save_latex_file_path)

            # Compile latex file to PDF. luatex may not be available on the system. Continue producing the tex files
            #  anyway, i.e., ignore the call's return value. Two runs are necessary for tikzmark references.
            compile_latex(save_latex_file_path, runs=2)

        print("Saved plot to", save_plot_file_name, "(" + save_plot_file_path + ")")
        # plt.close(fig)
    print("Time after visualization:", time.time() - start_time, "s")
    return list_of_gradient_values_as_array, sub_visualizations


def add_text_pdf_page(pdf_pages, text):
    def make_text_only(axis_obj):
        # Remove default axes and ticks to create a blank page:
        axis_obj.figure.clf()
        # Add text aligned to middle of figure:
        axis_obj.figure.text(0.5, 0.5, text, transform=axis_obj.figure.transFigure, ha="center")

    wrap_in_fig(pdf_pages, make_text_only)


def wrap_in_fig(pdf_pages, plot_func):
    fig = plt.figure(dpi=300, figsize=(15.27, 11.69))
    fig.tight_layout(rect=[0, 0, 0.5, 1])
    axis_obj = fig.add_subplot(1, 1, 1)
    plot_func(axis_obj)
    pdf_pages.savefig(fig)
    plt.close(fig)


def make_sub_plot_pdfpage_matplotlib(pdf_pages, *args, **kwargs):
    wrap_in_fig(pdf_pages, lambda axis_obj: make_sub_plot_matplotlib(axis_obj, *args, **kwargs))


def make_sub_plot_matplotlib(axes_obj, input_array, gradient_array, title, add_stats=True,
                             x_tick_positions=None, x_tick_labels=None,
                             y_tick_positions=None, y_tick_labels=None, y_tick_attributes=None, color_map=None):
    if x_tick_positions is None:
        x_tick_positions = range(input_array.shape[1])
    if x_tick_labels is None:
        x_tick_labels = x_tick_positions
    if y_tick_positions is None:
        y_tick_positions = range(input_array.shape[0])
    if y_tick_labels is None:
        y_tick_labels = y_tick_positions
    font_size_to_use = "x-small"

    # Determine min and max values for color scale
    vmax = np.percentile(np.abs(gradient_array), 95)  # np.max(np.abs(gradient_array))
    vmin = -vmax  # to make sure that black corresponds to zero
    stats = {"plot_min": vmin, "plot_max": vmax, "min": np.min(gradient_array),
             "max": np.max(gradient_array),
             "mean": np.mean(gradient_array),
             "std": np.std(gradient_array)}
    axes_obj.imshow(gradient_array, cmap=color_map, interpolation='nearest', vmin=vmin, vmax=vmax, aspect="auto")
    axes_obj.grid(False)  # ax1.grid(color="lightgreen")

    # Title
    axes_obj.set_title(title + (("\n" + str(stats).replace(",", ",\n")) if add_stats else ""), fontsize="small")

    # X-Axis-Ticks
    axes_obj.set_xticks(x_tick_positions)
    axes_obj.set_xticklabels(x_tick_labels, horizontalalignment='center', rotation=90,
                             color='green', fontsize=font_size_to_use)

    # Second y-axis with feature vectors(does not work):
    # ax2 = ax1.twiny()
    # ax2.set_yticks(np.arange(0, len(input_as_sequence_of_feat_vec_as_array), 1))
    # ax2.set_yticklabels(input_as_sequence_of_feat_vec_as_array, color='black')

    # Print current input / feature vectors into heat map:
    for k in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            curr_value = input_array[k, j]
            # Do not print zero decimals:
            if np.isnan(curr_value):
                print("Warning: NaN", k, j)
                curr_value = "NaN"
            elif int(curr_value) == curr_value:
                # Convert to int:
                curr_value = int(curr_value)
            else:
                curr_value = "{:.1f}".format(curr_value)
            axes_obj.text(j, k, curr_value, ha="center", va="center", color="green", fontsize=font_size_to_use)

            # Mark cells with non-zero values
            if gradient_array[k, j] != 0:
                side_length = 0.9  # 1.0 probably corresponds to whole heatmap square
                marker = plt.Rectangle((j - side_length / 2, k - side_length / 2), width=side_length,
                                       height=side_length, color='black', fill=False)
                axes_obj.add_artist(marker)

    # Y-Axis-Ticks
    axes_obj.yaxis.tick_right()
    axes_obj.yaxis.set_label_position("right")

    axes_obj.set_yticks(y_tick_positions)
    axes_obj.set_yticklabels(y_tick_labels, color='green', fontsize=font_size_to_use)
    if y_tick_attributes is not None:
        for tick_label, tick_attribute in zip(axes_obj.yaxis.get_ticklabels(), y_tick_attributes):
            for a_name, a_value in tick_attribute.items():
                set_attr_func = getattr(tick_label, "set_" + a_name, None)
                if set_attr_func is not None:
                    set_attr_func(a_value)
                # else:
                #     print("Warning: No set function/attribute for", a_name)


def make_sub_plot_bokeh(axes_obj, input_array, gradient_array, title, add_stats=True,
                        x_tick_positions=None, x_tick_labels=None,
                        y_tick_positions=None, y_tick_labels=None, y_tick_attributes=None, color_map=None):
    if x_tick_positions is None:
        x_tick_positions = range(input_array.shape[1])
    if x_tick_labels is None:
        x_tick_labels = x_tick_positions
    if y_tick_positions is None:
        y_tick_positions = range(input_array.shape[0])
    if y_tick_labels is None:
        y_tick_labels = y_tick_positions
    font_size_to_use = "x-small"

    # Determine min and max values for color scale
    vmax = np.percentile(np.abs(gradient_array), 95)  # np.max(np.abs(gradient_array))
    vmin = -vmax  # to make sure that black corresponds to zero

    if add_stats:
        stats = {"plot_min": vmin, "plot_max": vmax, "min": np.min(gradient_array),
                 "max": np.max(gradient_array),
                 "mean": np.mean(gradient_array),
                 "std": np.std(gradient_array)}
        title += "\n" + str(stats).replace(",", ",\n")

    from bokeh.io import show
    from bokeh.models import ColumnDataSource, LinearColorMapper
    from bokeh.plotting import figure

    scalar_to_cmap = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                                       cmap=color_map)

    arr = gradient_array
    x, y, val, text, color = [], [], [], [], []
    for row in range(gradient_array.shape[0]):
        for col in range(gradient_array.shape[1]):
            x.append(col + 0.5)
            y.append(row + 0.5)

            # Gradient color value:
            # Origin in bokeh is in lower left. Inverse y-axis:
            curr_value = gradient_array[gradient_array.shape[0] - row - 1, col]
            val.append(curr_value)  # not used atm
            color_tuple = scalar_to_cmap.to_rgba(curr_value, bytes=True)
            assert color_tuple[3] == 255, color_tuple
            color_tuple = color_tuple[0:3]
            color_str = "#"
            for color_num in color_tuple:
                color_str += "{:02X}".format(color_num)
            color.append(color_str)

            # Input value:
            # Origin in bokeh is in lower left. Inverse y-axis:
            curr_text = input_array[gradient_array.shape[0] - row - 1, col]
            # Do not print zero decimals:
            if int(curr_text) == curr_text:
                # Convert to int:
                curr_text = int(curr_text)
            else:
                curr_text = "{:.1f}".format(curr_text)
            text.append(curr_text)

    data = dict(x=x, y=y, val=val, text=text, color=color)
    print(data)
    source = ColumnDataSource(data)

    colors = ["blue", "lightblue", "white", "lightcoral", "red"]
    mapper = LinearColorMapper(palette=colors, low=vmin, high=vmax)

    p = figure(aspect_ratio=arr.shape[1] / arr.shape[0], title=title,
               x_range=[x_tick_labels[i] if i in x_tick_positions else "" for i in range(arr.shape[1])],
               # Origin in bokeh is in lower left. Inverse y-axis:
               y_range=[y_tick_labels[i] if i in y_tick_positions else "" for i in reversed(range(arr.shape[0]))],
               toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset,save",
               x_axis_location="below", y_axis_location="right",
               # Disable responsive design (i.e. do not size relative to browser window)
               sizing_mode='scale_width')

    p.rect(x="x", y="y", width=1, height=1, source=source,
           line_color=None, fill_color="color")  # transform('val', mapper))

    p.text(x="x", y="y", text="text", source=source, text_color="green", text_align="center", text_baseline="middle"
           )
    # color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
    #                      ticker=BasicTicker(desired_num_ticks=len(colors)),
    #                      formatter=PrintfTickFormatter(format="%d%%"))
    #
    # p.add_layout(color_bar, 'left')

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    # p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.yaxis.axis_label_text_font_style = "bold"

    show(p)
