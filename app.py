import os
import gradio as gr
import torch
import functools
from datasets import load_dataset
import importlib

# os.environ['HTTP_PROXY'] = 'http://192.168.48.17:18000'
# os.environ['HTTPS_PROXY'] = 'http://192.168.48.17:18000'

MAX_BATCH_SIZE = 16
MAX_QUEUE_SIZE = 10
MAX_K_RETRIEVAL = 20
cache_dir = "./.cache"
option = "uni3d"

module = importlib.import_module(f"feature_extractors.{option}_embedding_encoder")
encoder = getattr(module, f"{option.capitalize()}EmbeddingEncoder")(cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_id_list = torch.load("data/source_id_list.pt")
source_to_id = {source_id: i for i, source_id in enumerate(source_id_list)}
dataset = load_dataset("VAST-AI/LD-T3D", name=f"rendered_imgs_diag_above", split="base", cache_dir=cache_dir)

@functools.lru_cache()
def get_embedding(option, modality, angle=None):
    save_path = f'data/objaverse_{option}_{modality + (("_" + str(angle)) if angle is not None else "")}_embeddings.pt'
    if os.path.exists(save_path):
        return torch.load(save_path)
    else:
        return gr.Error(f"Embedding file not found: {save_path}")

def predict(xb, xq, top_k):
    xb = xb.to(xq.device)
    sim = xq @ xb.T # (nq, nb)
    _, indices = sim.topk(k=top_k, largest=True)
    return indices

def get_image(index):
    return dataset[index]["image"]

def retrieve_3D_models(textual_query, top_k, modality_list):
    if textual_query == "":
        raise gr.Error("Please enter a textual query")
    if len(textual_query.split()) > 20:
        gr.Warning("Retrieval result may be inaccurate due to long textual query")
    if len(modality_list) == 0:
        raise gr.Error("Please select at least one modality")
    
    def _retrieve_3D_models(query, top_k, modals:list):
        op = "add"
        is_text = True if "text" in modals else False
        is_3D = True if "3D" in modals else False
        if is_text:
            modals.remove("text")
        if is_3D:
            modals.remove("3D")
        angles = modals

        # get base embeddings
        embeddings = []
        if is_text:
            embeddings.append(get_embedding(option, "text"))
        if len(angles) > 0:
            for angle in angles:
                embeddings.append(get_embedding(option, "image", angle=angle))
        if is_3D:
            embeddings.append(get_embedding(option, "3D"))
            
        ## fuse base embeddings
        if len(embeddings) > 1:
            if op == "concat":
                embeddings = torch.cat(embeddings, dim=-1)
            elif op == "add":
                embeddings = sum(embeddings)
            else:
                raise ValueError(f"Unsupported operation: {op}")
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        else:
            embeddings = embeddings[0]

        # encode query embeddings
        xq = encoder.encode_query(query)
        if op == "concat":
            xq = xq.repeat(1, embeddings.shape[-1] // xq.shape[-1]) # repeat to be aligned with the xb
            xq /= xq.norm(dim=-1, keepdim=True)
        
        pred_ind_list = predict(embeddings, xq, top_k)
        return pred_ind_list[0].cpu().tolist() # we have only one query

    indices = _retrieve_3D_models(textual_query, top_k, modality_list)
    return [get_image(index) for index in indices]

def launch():
    with gr.Blocks() as demo:
        with gr.Row():
            textual_query = gr.Textbox(label="Textual Query", autofocus=True,
                                       placeholder="A chair with a wooden frame and a cushioned seat")
            modality_list = gr.CheckboxGroup(label="Modality List", value=[],
                                             choices=["text", "front", "back", "left", "right", "above", 
                                                      "below", "diag_above", "diag_below", "3D"])
        with gr.Row():
            top_k = gr.Slider(minimum=1, maximum=MAX_K_RETRIEVAL, step=1, label="Top K Retrieval Result", 
                              value=5, scale=2)
            run = gr.Button("Search", scale=1)
            clear_button = gr.ClearButton(scale=1)
        with gr.Row():
            output = gr.Gallery(format="webp", label="Retrieval Result", columns=5, type="pil")
        run.click(retrieve_3D_models, [textual_query, top_k, modality_list], output, 
                #   batch=True, max_batch_size=MAX_BATCH_SIZE
                  )
        clear_button.click(lambda: ["", 5, [], []], outputs=[textual_query, top_k, modality_list, output])
        examples = gr.Examples(examples=[["An ice cream with a cherry on top", 10, ["text", "front", "back", "left", "right", "above", "below", "diag_above", "diag_below", "3D"]],
                                         ["A mid-age castle", 10, ["text", "front", "back", "left", "right", "above", "below", "diag_above", "diag_below", "3D"]],
                                         ["A coke", 10, ["text", "front", "back", "left", "right", "above", "below", "diag_above", "diag_below", "3D"]]],
                            inputs=[textual_query, top_k, modality_list],
                            # cache_examples=True,
                            outputs=output,
                            fn=retrieve_3D_models)
        
    demo.queue(max_size=10)

    # os.environ.pop('HTTP_PROXY')
    # os.environ.pop('HTTPS_PROXY')

    demo.launch(server_name='0.0.0.0')

if __name__ == "__main__":
    launch()
    # print(len(retrieve_3D_models("A chair with a wooden frame and a cushioned seat", 5, ["3D", "diag_above", "diag_below"])))