import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer modeli tanımı
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # İlk katman: Kenarların korunması için reflection padding ve konvolüsyon
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = nn.Conv2d(3, 64, 7)  # Giriş: 3 kanal (RGB), Çıkış: 64 kanal, Çekirdek boyutu: 7x7
        self.in01_1 = InstanceNormalization(64)  # 64 kanallı instance normalizasyon

        # İkinci katman: Özelliklerin daha fazla çıkarılması
        self.conv02_1 = nn.Conv2d(64, 128, 3, 2, 1)  # Çıkış kanal sayısı: 128, Stride: 2, Padding: 1
        self.conv02_2 = nn.Conv2d(128, 128, 3, 1, 1)  # Aynı boyutta devam eden konvolüsyon
        self.in02_1 = InstanceNormalization(128)  # Instance normalization uygulanır

        # Üçüncü katman: Daha fazla kanal ve detay
        self.conv03_1 = nn.Conv2d(128, 256, 3, 2, 1)  # Çıkış kanal sayısı: 256
        self.conv03_2 = nn.Conv2d(256, 256, 3, 1, 1)  # Aynı boyutta devam eden konvolüsyon
        self.in03_1 = InstanceNormalization(256)

        # Residual (artık) bloklar: 256 kanallı 8 adet ardışık blok
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(8)]
        )

        # İlk çözümleyici katman: Kanal sayısını azaltarak boyut büyütme
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)  # Çıkış kanal sayısı: 128
        self.deconv01_2 = nn.Conv2d(128, 128, 3, 1, 1)  # Konvolüsyon uygulanır
        self.in12_1 = InstanceNormalization(128)

        # İkinci çözümleyici katman: Kanal sayısını daha da azaltma
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)  # Çıkış kanal sayısı: 64
        self.deconv02_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)

        # Son katman: Orijinal boyutlara dönme
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = nn.Conv2d(64, 3, 7)  # Çıkış: 3 kanal (RGB)

    def forward(self, x):
        # Girişten ilk özellik çıkarımı
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        # Daha derin özellik çıkarımı
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        y = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))

        # Residual bloklardan geçiş
        y = self.res_blocks(y)

        # Çözümleyici katmanlarla orijinal boyutlara dönme
        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = torch.tanh(self.deconv03_1(self.refpad12_1(y)))  # Çıkış için tanh aktivasyonu

        return y

# Residual (artık) blok tanımı
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        # Residual blok içerisindeki ardışık işlemler
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),  # Çıkış kanal sayısı sabit (dim)
            InstanceNormalization(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            InstanceNormalization(dim),
        )

    def forward(self, x):
        # Girişe, işlenmiş sonucu ekleyerek geri döndür
        return x + self.block(x)

# Instance Normalization tanımı
class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))  # Ölçek parametresi (öğrenilebilir)
        self.shift = nn.Parameter(torch.FloatTensor(dim))  # Kaydırma parametresi (öğrenilebilir)
        self.eps = eps  # Sayısal kararlılık için küçük epsilon
        self._reset_parameters()  # Parametreleri başlat

    def _reset_parameters(self):
        self.scale.data.uniform_()  # Ölçek parametresini rastgele başlat
        self.shift.data.zero_()  # Kaydırma parametresini sıfırla

    def forward(self, x):
        # Özellik haritalarının ortalama ve varyansını hesapla
        mean = x.mean([2, 3], keepdim=True)
        var = x.var([2, 3], keepdim=True, unbiased=False)
        # Normalizasyon işlemini uygula
        return (x - mean) / torch.sqrt(var + self.eps) * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)