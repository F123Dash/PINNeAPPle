"""PINNeAPPle zoo demo (Hugging Face Space): every zoo model, loaded from the Hub, with its physics checks.

UI only: the predictions, sweeps, checks, plots and report come from ``pinneapple_hub.zoo.demo``. The Space is
published by ``scripts/zoo/publish_to_hub.py --space`` (``.github/workflows/publish-zoo.yml``); locally:

    pip install -e ".[hub]" gradio && python apps/zoo_space/app.py
"""
from __future__ import annotations

import os

# load every model from its Hub repo (SHA-256 checked), as a user of the zoo would, not from the package's copy
os.environ.setdefault("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")

import gradio as gr  # noqa: E402

from pinneapple_hub.zoo import demo  # noqa: E402

MAX_INPUTS = 16
INTRO = ("Pick a zoo model, set its inputs and the input to sweep. The plot shows the prediction along the sweep "
         "with the ensemble spread or the conformal bound; shaded parts are where the model's query checks warn "
         "(yellow) or fail (red: outside the validity range, an extrapolation). The table gives every check for the "
         "chosen input. Models are downloaded from the [PINNeAPPle-Labs](https://huggingface.co/PINNeAPPle-Labs) "
         "repos; the code is [PINNeAPPle](https://github.com/PINNeAPPle-Labs/PINNeAPPle) (`pinneapple_hub.zoo`).")


def _box(d=None):
    if d is None:
        return {"visible": False, "value": None, "label": "", "info": None}
    return {"visible": True, "value": d["value"], "label": f"{d['name']} [{d['unit']}] ({d['lo']:g} … {d['hi']:g})",
            "info": d["description"]}


def _fields(name):
    c = demo.controls(name)
    names = [d["name"] for d in c]
    boxes = [gr.update(**_box(d)) for d in c] + [gr.update(**_box()) for _ in range(MAX_INPUTS - len(c))]
    return [gr.update(choices=names, value=names[-1]), *boxes]


def _run(name, vary, *values):
    c = demo.controls(name)
    inputs = {d["name"]: (d["value"] if v is None else float(v)) for d, v in zip(c, values, strict=False)}
    r = demo.sweep(name, inputs, vary if vary in inputs else c[-1]["name"])
    return demo.figure(r), demo.report(r)


def build() -> gr.Blocks:
    first = demo.models()[0]
    c = demo.controls(first)
    with gr.Blocks(title="PINNeAPPle zoo demo") as app:
        gr.Markdown("# PINNeAPPle zoo: predictions with their physics checks\n\n" + INTRO)
        with gr.Row():
            model = gr.Dropdown(demo.models(), value=first, label="Model")
            vary = gr.Dropdown([d["name"] for d in c], value=c[-1]["name"], label="Sweep this input",
                               allow_custom_value=True)          # the choices change with the model
        with gr.Row():
            with gr.Column(scale=1):
                boxes = [gr.Number(**_box(c[i] if i < len(c) else None)) for i in range(MAX_INPUTS)]
                go = gr.Button("Predict", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Sweep")
                text = gr.Markdown()
        model.change(_fields, model, [vary, *boxes]).then(_run, [model, vary, *boxes], [plot, text])
        go.click(_run, [model, vary, *boxes], [plot, text])
        app.load(_run, [model, vary, *boxes], [plot, text])
    return app


if __name__ == "__main__":
    build().launch()
