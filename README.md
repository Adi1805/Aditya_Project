# **Neural Style Transfer with PyTorch**
Repaint any photograph in the brushstrokes of a painting you love — powered by a VGG19 network that was never trained to do this in the first place.

<img width="116" height="31" alt="image" src="https://github.com/user-attachments/assets/36eea6d9-2c7c-4039-afc1-b4ba8ebef83d" />
<img width="100" height="29" alt="image" src="https://github.com/user-attachments/assets/9bef39a2-c883-4705-a030-0b73bd1cccd5" />
<img width="132" height="31" alt="image" src="https://github.com/user-attachments/assets/90fda805-e942-4ae4-bca2-653c92b5fb55" />

What this is

This project takes a content image (say, a photo you took) and a style image (say, a Van Gogh painting), and generates a brand new image that keeps the structure of the first and borrows the texture, color, and brushwork of the second.

There's no dataset here, no training loop in the traditional sense, and no dedicated "style transfer network." That's what makes the algorithm interesting — it repurposes a classifier (VGG19, trained on ImageNet to recognize dogs and cars) and uses it as a feature extractor to measure two things: how similar is the content and how similar is the texture. Then it optimizes the pixels of an image directly until both losses are minimized.
