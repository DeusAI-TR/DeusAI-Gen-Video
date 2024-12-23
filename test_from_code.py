import time  # Zaman ölçümü için gerekli modül
import os  # Dosya işlemleri için gerekli modül

import numpy as np  # Matematiksel işlemler ve dizi manipülasyonu için
from PIL import Image  # Görüntü işleme için

import torch  # Derin öğrenme işlemleri için
import torchvision.transforms as transforms  # Görüntü dönüşümleri için
from torch.autograd import Variable  # PyTorch değişkenlerini tanımlamak için

# Görüntü dönüştürme ve stil transfer işlemi gerçekleştiren fonksiyon
def transform(models, style, input, load_size=450, gpu=-1):
    # Seçilen stildeki modeli al
    model = models[style]

    # Eğer GPU kullanılacaksa modeli CUDA'ya taşı
    if gpu > -1:
        model.cuda()
    else:
        model.float()  # GPU yoksa modeli float tipine çevir

    # Girdi görüntüsünü RGB formatında yükle
    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size  # Görüntü boyutlarını al

    # Görüntü oranını hesapla
    ratio = h * 1.0 / w

    # Görüntü boyutlarını oran koruyarak yeniden boyutlandır
    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    # Görüntüyü yeniden boyutlandır ve numpy dizisine çevir
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    # Görüntünün renk kanallarını BGR'den RGB'ye çevir
    input_image = input_image[:, :, [2, 1, 0]]
    # Görüntüyü tensöre çevir ve 4 boyutlu hale getir
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # Tensörü [-1, 1] aralığına ölçekle
    input_image = -1 + 2 * input_image

    # Tensörü GPU'ya ya da float tipine taşı
    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    # Zaman ölçümünü başlat
    t0 = time.time()
    print("input shape", input_image.shape)  # Girdi tensörünün boyutunu yazdır

    # Model ile tahmin yap
    with torch.no_grad():  # Gradientsiz çalışarak performansı artır
        output_image = model(input_image)[0]

    print("inference time took ", {time.time() - t0}, " s")  # İşlem süresini yazdır

    # Çıktı görüntüsünün renk kanallarını BGR'den RGB'ye çevir
    output_image = output_image[[2, 1, 0], :, :]
    # Çıktıyı [0, 1] aralığına ölçekle
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    # Çıktıyı numpy dizisine çevir
    output_image = output_image.numpy()

    # Çıktıyı görüntü formatına (HWC) çevir ve 0-255 aralığına ölçekle
    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    # Çıktıyı bir görüntü nesnesine çevir
    output_image = Image.fromarray(output_image)

    # İşlenmiş görüntüyü döndür
    return output_image