import torch
import torch.nn.functional as F
import clip

class ClipCriteria:
    def __init__(self, device):
        self.device = deivce
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)


    def clipI(self, image_1, image_2):
        image_1 = self.preprocess(image_1).unsqueeze(0).to(device)
        image_2 = self.preprocess(image_2).unsqueeze(0).to(device)

        embedding_1 = self.model.encode_image(image_2)
        embedding_2 = self.model.encode_image(image_2)

        return F.cosine_similarity(embedding_1, embedding_2)

    def clipT(self, image, text_prompt):
        image = self.preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize([text_prompt]).to(device)

        score = self.model(image, text)
        return score
