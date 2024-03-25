import gradio as gr
from PIL import Image

MAX_BATCH_SIZE = 16
MAX_QUEUE_SIZE = 10
MAX_K_RETRIEVAL = 20

def retrieve_3D_models(textual_query, top_k, modality_list):
    if textual_query == "":
        gr.Error("Please enter a textual query")
    if len(textual_query) > 20:
        gr.Warning("Retrieval result may be inaccurate due to long textual query")
    if len(textual_query) > 77:
        gr.Error("Textual query is too long")
    if len(modality_list) == 0:
        gr.Error("Please select at least one modality")
    return [Image.new("RGB", (224, 224))]*5


def launch():
    with gr.Blocks() as demo:
        with gr.Row():
            textual_query = gr.Textbox(label="Textual Query", placeholder="A chair with a wooden frame and a cushioned seat")
            modality_list = gr.CheckboxGroup(label="Modality List", value=[],
                                             choices=["text", "front", "back", "left", "right", "above", "below", "diag_above", "diag_below", "3D"])
        with gr.Row():
            top_k = gr.Slider(minimum=1, maximum=MAX_K_RETRIEVAL, step=1, label="Top K Retrieval Result", value=5, scale=2)
            run = gr.Button("Search", scale=1)
            clear_button = gr.ClearButton(scale=1)
        with gr.Row():
            output = gr.Gallery(format="webp", label="Retrieval Result", columns=5, type="pil")
        run.click(retrieve_3D_models, [textual_query, top_k, modality_list], output
                #   , batch=True, max_batch_size=MAX_BATCH_SIZE
                  )
        clear_button.click(lambda: ["", 5, [], []], outputs=[textual_query, top_k, modality_list, output])
        # examples = gr.Examples(examples=[["An ice cream with a cherry on top", 10, ["3D", "diag_above", "diag_below"]]],
        #                     inputs=[textual_query, top_k, modality_list],
        #                    cache_examples=True,
        #                    outputs=output,
        #                    fn=retrieve_3D_models)
        
    demo.queue(max_size=10)
    demo.launch()

if __name__ == "__main__":
    launch()