import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import function as fn

# Entrada da imagem 
INPUT_FILE = 'foto1.jpeg'
OUTPUT_GRAY_FILE = 'foto1Cinza.png' # arquivo de saida da imagem cinza
OUTPUT_FILE = 'foto1Processada.png' # arquivo de saida da imagem procesada

matrizBase = np.array([
    [ 0.299000,  0.587000,  0.114000],
    [ 0.500000, -0.418688, -0.081312],
    [-0.168736, -0.331264,  0.500000]
], dtype=np.float64)

deslocamento = np.array([0, 128, 128], dtype=np.float64)

ycrcb = fn.rgb_to_ycrcb(INPUT_FILE, OUTPUT_GRAY_FILE, matrizBase, deslocamento)

# 2 Versão A - DCT 2D pela operação matricial

# 2.2 DCT 2D

F_blocos, blocos, Y, numBlocosAltura, numBlocosLargura, altura, largura = fn.apply_dct(ycrcb, 8)

print("Formato dos blocos transformados:", F_blocos.shape)
print("Primeiro bloco antes da DCT:")
print(blocos[0, 0])
print("Primeiro bloco depois da DCT:")
print(F_blocos[0, 0])

# 3 Quantização

Q = np.array([
[16, 11, 10, 16, 24, 40, 51, 61],
[12, 12, 14, 19, 26, 58, 60, 55],
[14, 13, 16, 24, 40, 57, 69, 56],
[14, 17, 22, 29, 51, 87, 80, 62],
[18, 22, 37, 56, 68, 109, 103, 77],
[24, 35, 55, 64, 81, 104, 113, 92],
[49, 64, 78, 87,103, 121, 120, 101],
[72, 92, 95, 98,112, 100, 103, 99]
], dtype=np.float64)

print("Matriz de Quantização (Q):")
print(Q)

# Fator de escala
s = [0.5, 1, 2, 4]


Fq_blocos, Qs = fn.quantization(F_blocos, Q, s[1], numBlocosAltura, numBlocosLargura)

print("Primeiro bloco quantizado:")
print(Fq_blocos[0, 0])

# Parte 4  Varredura Zig-Zag

zigzag_blocos, ordem_zigzag = fn.zigZag_Scan(8, Fq_blocos, numBlocosAltura, numBlocosLargura)

print("Primeiras posições zig-zag:")
print(ordem_zigzag[:10])

print("Primeiro bloco em ordem zig-zag:")
print(zigzag_blocos[0, 0])

# Parte 5  Codificação RLE

# 5.1 RLE mínimo sugerido

rle_blocos = fn.rle(numBlocosAltura, numBlocosLargura, zigzag_blocos)

print("Codificação RLE do primeiro bloco:")
print(rle_blocos[0][0])

# 5.2 Bits por pixel
# Contagem dos bits da representação RLE


totalBits, totalPares, totalEOB = fn.counter_rle(rle_blocos)

print("Total de pares:", totalPares)
print("Total de marcadores EOB:", totalEOB)
print("Total de bits comprimidos:", totalBits)

totalPixels = altura * largura
bpp = totalBits / totalPixels

print("Total de pixels:", totalPixels)
print("BPP da imagem:", bpp)
print("BPP original sem compressão: 8")

# 6 Procedimento inverso

# 6.1 Desquantização

# Parte 6.1 - Decodificação RLE

vetores_recuperados = fn.decode_rle(rle_blocos, numBlocosAltura, numBlocosLargura)

print("Vetor recuperado do primeiro bloco:")
print(vetores_recuperados[0][0])
print("Quantidade de elementos:", len(vetores_recuperados[0][0]))


Fq_recuperados = fn.reconstruct_Fq(vetores_recuperados, ordem_zigzag, numBlocosAltura, numBlocosLargura)

print("Matriz recuperada do primeiro bloco:")
print(Fq_recuperados[0, 0])


# 6.2 IDCT matricial


fq_desquantizados = fn.apply_idct(Fq_recuperados, Q, s[1], numBlocosAltura, numBlocosLargura)


blocos_reconstruidos = fn.reconstruct_blocks(fq_desquantizados, fn.matrix_dct(blockSize=8), numBlocosAltura, numBlocosLargura)


Y_reconstruida = fn.reconstruct_image(blocos_reconstruidos, numBlocosAltura, numBlocosLargura, altura, largura, OUTPUT_FILE)

# 7 Métricas de qualidade
# 7.1 PSNR

original = Y.astype(np.float64)
reconstruida = Y_reconstruida.astype(np.float64)

MSE, PSNR = fn.mse_psnr(original, reconstruida)

print("MSE:", MSE)
print("PSNR:", PSNR, "dB")

#7.2 SSIM

SSIM = structural_similarity(
    Y,
    Y_reconstruida,
    data_range=255
)

print("SSIM:", SSIM)

# 8 Curva qualidade × compressão
