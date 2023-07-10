import gradio as gr
from PIL import Image

from predict_pipeline import PredictPipeline


def greet(img1, img2):
    img1_pil = Image.fromarray(img1).convert('RGB')
    img2_pil = Image.fromarray(img2).convert('RGB')
    pipeline = PredictPipeline()
    pred, df = pipeline.predict_proba(img1_pil, img2_pil)
    return df, int(pred > 0.4)



demo = gr.Interface(fn=greet, inputs=["image", "image"], outputs=[gr.Dataframe(row_count=(1, "dynamic"),
                                                                               col_count=(18, "dynamic"),
                                                                               label="Features"
                                                                               ),
                                                                  "label"])


demo.launch()
