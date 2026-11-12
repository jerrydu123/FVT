# Third-party code and publication metadata

The extracted `dinounet/dinov3/` files retain their Meta copyright and DINOv3 License Agreement headers. They were copied from the local implementation already used by this project. Package initializers were reduced to avoid importing unrelated training frameworks. Do not remove the source copyright headers.

The complete official DINOv3 license is included in [LICENSES/DINOv3-LICENSE.md](../LICENSES/DINOv3-LICENSE.md). Its source URL, retrieval date, and SHA-256 are recorded in LICENSES/SOURCES.json. The retrieved agreement is dated August 19, 2025. Existing source headers are preserved. This project does not relicense the DINOv3 materials under MIT or bundle pretrained weights. Distribution of DINOv3 materials and their derivatives remains subject to that agreement, including applicable terms for checkpoint weights containing DINOv3 parameters.

The project also uses PyTorch, torchvision (including ResNet-34), NumPy, SciPy, scikit-image, Pillow, OpenCV, Matplotlib, and tensorboardX as external dependencies. The provenance and redistribution permissions for your existing project-specific files under `utils/`, `models/`, and the DINOv3 adapter should be confirmed by the author.

The intended license for original project contributions is MIT. At the author's request, copyright attribution and the final root LICENSE are deferred. No completed MIT grant is included in this release candidate; the local draft is kept outside the publication directory. Third-party materials remain under their applicable licenses and notices. Datasets are distributed by their original providers under their own terms; source publication does not relicense them.
