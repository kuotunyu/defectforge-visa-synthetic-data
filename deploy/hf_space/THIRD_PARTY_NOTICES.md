# Third-party notices

| Component | Revision | License | Use in this Space |
|---|---|---|---|
| `timm/convnext_tiny.fb_in1k` | `b43a6303c9fcf176d2d707478a128c2c91e93528` | Apache-2.0 | Classifier architecture and pretrained initialization |
| `nvidia/segformer-b0-finetuned-ade-512-512` | `489d5cd81a0b59fab9b7ea758d3548ebe99677da` | NVIDIA Source Code License for SegFormer | Segmenter architecture and pretrained initialization |
| VisA | VisA 2022 release | CC BY 4.0 | Training/evaluation source; five unmodified examples are shipped for interactive Demo use |

The SegFormer license limits the Work and derivative works to **non-commercial
research or evaluation use**. The complete upstream license is included at
`licenses/NVIDIA_SEGFORMER_LICENSE.txt`.

The Apache-2.0 text applicable to the ConvNeXt base model is included at
`licenses/APACHE-2.0.txt`.

VisA attribution:

> Zou, Y., Jeong, J., Pemula, L., Zhang, D., and Dabeer, O. SPot-the-Difference
> Self-supervised Pre-training for Anomaly Detection and Segmentation. ECCV 2022.

- VisA: https://registry.opendata.aws/visa/
- Included example inventory and SHA256: `examples/README.md`
- ConvNeXt model card: https://huggingface.co/timm/convnext_tiny.fb_in1k
- SegFormer model card: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
