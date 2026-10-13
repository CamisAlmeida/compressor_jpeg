import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

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

# Recebe uma imagem de entrada, converte para YCrCb e salva a imagem em tons de cinza
def rgb_to_ycrcb(input, output, matrizBase, deslocamento):
    # 1 Requisitos de entrada e preparação
    # 1.1 conversão RGB → YCrCb 
    # Abrir a imagem
    img = Image.open(input).convert('RGB')

    # Converter a imagem em uma matriz NumPy
    rgb = np.array(img, dtype=np.float64)

    # YCrCb 
    ycrcb = rgb @ matrizBase.T + deslocamento

    # Salvar a imagem em tons de cinza

    # Arredondar os valores e converter para 8 bits
    imagemCinza = np.clip(np.round(ycrcb[:, :, 0]), 0, 255).astype(np.uint8)

    # Criar e salvar a imagem
    imgCinza = Image.fromarray(imagemCinza)
    imgCinza.save(output)


    print("Dimensões da imagem:", ycrcb.shape)
    print("Tipo dos dados:", ycrcb.dtype)
    print("Primeiro pixel de YCrCb:", ycrcb[0, 0])
    return ycrcb


ycrcb = rgb_to_ycrcb(INPUT_FILE, OUTPUT_GRAY_FILE, matrizBase, deslocamento)


def divide_blocks(ycrcb, block_size=8):
    # Parte 1.2 Deslocamento de nível

    # Extrair o canal Y (luminância)
    Y = ycrcb[:, :, 0]

    #Subtrair 128 do canal Y
    imagemDeslocada = Y - 128

    print("primeiro pixel da imagem deslocada:", imagemDeslocada[0, 0])

    # Parte 1.3 Divisão em blocos 8x8

    # Dimensões da imagem deslocada
    altura, largura = imagemDeslocada.shape

    # Quantas linhas e colunas faltam para múltiplos de 8
    paddingAltura = (8 - altura % 8) % 8
    paddingLargura = (8 - largura % 8) % 8

    print("Dimensões originais:", imagemDeslocada.shape)
    print("Linhas a adicionar:", paddingAltura)
    print("Colunas a adicionar:", paddingLargura)

    # Adicionar padding nas bordas, se necessário
    imagemPadded = np.pad(
        imagemDeslocada,
        ((0, paddingAltura), (0, paddingLargura)),
        mode='edge'
    )

    print("Dimensões após o padding:", imagemPadded.shape)

    # Calcular a quantidade de blocos em cada direção
    numBlocosAltura = imagemPadded.shape[0] // 8
    numBlocosLargura = imagemPadded.shape[1] // 8

    # Dividir a imagem em blocos 8x8
    blocos = imagemPadded.reshape(
        numBlocosAltura, 8,
        numBlocosLargura, 8
    ).transpose(0, 2, 1, 3)

    print("Quantidade de blocos na vertical:", numBlocosAltura)
    print("Quantidade de blocos na horizontal:", numBlocosLargura)
    print("Formato da matriz de blocos:", blocos.shape)
    print("Primeiro bloco 8x8:")
    print(blocos[0, 0])

    return blocos, numBlocosAltura, numBlocosLargura, altura, largura
    

# 2 Versão A - DCT 2D pela operação matricial

def matrix_dct(blockSize=8):
        
    # 2.1 Construção da matriz da DCT
    N = blockSize

    # Cria uma matriz 8x8 
    C = np.zeros((N, N), dtype=np.float64)

    # Calcula cada elemento da matriz
    for k in range(N):
        for n in range(N):

            # Fator de normalização
            if k == 0:
                alfa = math.sqrt(1 / N)
            else:
                alfa = math.sqrt(2 / N)

            # Elemento da matriz DCT
            C[k, n] = alfa * math.cos(
                (math.pi / N) * (n + 0.5) * k
            )

    print("Matriz da DCT (8x8):")
    print(C)

    print("Dimensões da matriz:", C.shape)
    return C

# 2.2 DCT 2D

def apply_dct(ycrcb, blockSize=8):
    blocos, numBlocosAltura, numBlocosLargura, altura, largura = divide_blocks(ycrcb)
    C = matrix_dct(blockSize=8)
    F_blocos = np.zeros_like(blocos, dtype=np.float64)

    # Aplicar a DCT 2D em cada bloco
    for i in range(numBlocosAltura):
        for j in range(numBlocosLargura):
            bloco = blocos[i, j]
            F_blocos[i, j] = C @ bloco @ C.T
    return F_blocos, blocos, numBlocosAltura, numBlocosLargura, altura, largura
   
F_blocos, blocos, numBlocosAltura, numBlocosLargura, altura, largura = apply_dct(ycrcb, blockSize=8)

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

def quantization(F_blocos, Q, s):

    Qs = s * Q # Intensidade de quntização

    Fq_blocos = np.zeros_like(F_blocos, dtype=np.int32)

    # Percorre todos os blocos
    for i in range(numBlocosAltura):
        for j in range(numBlocosLargura):

            # Coeficientes DCT do bloco
            F = F_blocos[i, j]
            Fq = np.round(F / Qs)
            Fq_blocos[i, j] = Fq.astype(np.int32)
    return Fq_blocos, Qs


Fq_blocos, Qs = quantization(F_blocos, Q, s[1]) 

print("Primeiro bloco quantizado:")
print(Fq_blocos[0, 0])

# Parte 4  Varredura Zig-Zag
def zigZag_Scan(blockSize, Fq_blocos, ordem_zigzag):
    ordem_zigzag = []

    N = blockSize

    for diagonal in range(2 * N - 1):
        linha_min = max(0, diagonal - (N - 1))
        linha_max = min(N - 1, diagonal)

        if diagonal % 2 == 0:
            linhas = range(linha_max, linha_min - 1, -1)
        else:
            linhas = range(linha_min, linha_max + 1)

        for linha in linhas:
            coluna = diagonal - linha
            ordem_zigzag.append((linha, coluna))

    # Armazena as sequências zig-zag de todos os blocos
    zigzag_blocos = np.zeros(
        (numBlocosAltura, numBlocosLargura, 64),
        dtype=np.int32
    )

    # Percorre todos os blocos quantizados
    for i in range(numBlocosAltura):
        for j in range(numBlocosLargura):

            Fq = Fq_blocos[i, j]

            # Reorganiza os 64 coeficientes na ordem zig-zag
            for k, (linha, coluna) in enumerate(ordem_zigzag):
                zigzag_blocos[i, j, k] = Fq[linha, coluna]

    return zigzag_blocos, ordem_zigzag

zigzag_blocos, ordem_zigzag = zigZag_Scan(8, Fq_blocos, ordem_zigzag=[])

print("Primeiras posições zig-zag:")
print(ordem_zigzag[:10])

print("Primeiro bloco em ordem zig-zag:")
print(zigzag_blocos[0, 0])

# Parte 5  Codificação RLE

# 5.1 RLE mínimo sugerido
def rle(heightBlocks, widthBlocks, zigzag_Blocks):
    rle_blocks = []

    for i in range(heightBlocks):
        linha_rle = []

        for j in range(widthBlocks):
            vetor = zigzag_Blocks[i, j]

            run = 0
            codificacao = []

            for valor in vetor:
                if valor == 0:
                    run += 1
                else:
                    codificacao.append((run, int(valor)))
                    run = 0

            # Finaliza a codificação do bloco
            codificacao.append("EOB")

            linha_rle.append(codificacao)

        rle_blocks.append(linha_rle)
    return rle_blocks

rle_blocos = rle(numBlocosAltura, numBlocosLargura, zigzag_blocos)

print("Codificação RLE do primeiro bloco:")
print(rle_blocos[0][0])

# 5.2 Bits por pixel
# Contagem dos bits da representação RLE

def counter_rle(rle_blocks):

    totalBits = 0
    totalPares = 0
    totalEOB = 0

    for line_rle in rle_blocks:
        for codificacao in line_rle:
            # Cada item é um par ou o marcador EOB
            pares = len(codificacao) - 1

            totalPares += pares
            totalEOB += 1

            totalBits += pares * 24
            totalBits += 8  # EOB
    return totalBits, totalPares, totalEOB

totalBits, totalPares, totalEOB = counter_rle(rle_blocos)

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
vetores_recuperados = []

for i in range(numBlocosAltura):
    linha_vetores = []

    for j in range(numBlocosLargura):
        codificacao = rle_blocos[i][j]
        vetor = []

        for item in codificacao:
            if item == "EOB":
                break

            run, valor = item

            # Recupera os zeros anteriores ao valor
            vetor.extend([0] * run)

            # Recupera o coeficiente não nulo
            vetor.append(valor)

        # Completa os zeros que faltam até 64 posições
        vetor.extend([0] * (64 - len(vetor)))

        linha_vetores.append(vetor)

    vetores_recuperados.append(linha_vetores)

print("Vetor recuperado do primeiro bloco:")
print(vetores_recuperados[0][0])
print("Quantidade de elementos:", len(vetores_recuperados[0][0]))


Fq_recuperados = np.zeros(
    (numBlocosAltura, numBlocosLargura, 8, 8),
    dtype=np.int32
)

for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):
        vetor = vetores_recuperados[i][j]

        for k, (linha, coluna) in enumerate(ordem_zigzag):
            Fq_recuperados[i, j, linha, coluna] = vetor[k]

print("Matriz recuperada do primeiro bloco:")
print(Fq_recuperados[0, 0])


# 6.2 IDCT matricial

Qs = s[1] * Q

F_desquantizados = np.zeros(
    (numBlocosAltura, numBlocosLargura, 8, 8),
    dtype=np.float64
)

for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):
        F_desquantizados[i, j] = Fq_recuperados[i, j] * Qs

print("Coeficientes desquantizados do primeiro bloco:")
print(F_desquantizados[0, 0])

blocos_reconstruidos = np.zeros(
    (numBlocosAltura, numBlocosLargura, 8, 8),
    dtype=np.float64
)

for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):
        F = F_desquantizados[i, j]

        blocos_reconstruidos[i, j] = C.T @ F @ C

#Desfazer o deslocamento
blocos_reconstruidos += 128

blocos_reconstruidos = np.clip(
    np.rint(blocos_reconstruidos),
    0,
    255
).astype(np.uint8)

#Montagem da imagem reconstruída
altura_padded = numBlocosAltura * 8
largura_padded = numBlocosLargura * 8

Y_reconstruida_padded = np.zeros(
    (altura_padded, largura_padded),
    dtype=np.uint8
)

for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):
        inicio_linha = i * 8
        inicio_coluna = j * 8

        Y_reconstruida_padded[
            inicio_linha:inicio_linha + 8,
            inicio_coluna:inicio_coluna + 8
        ] = blocos_reconstruidos[i, j]

# Remover o padding e recuperar o tamanho original
Y_reconstruida = Y_reconstruida_padded[:altura, :largura]

# Criar e salvar a imagem
imgCinza = Image.fromarray(Y_reconstruida)
imgCinza.save(OUTPUT_FILE)

# 7 Métricas de qualidade
# 7.1 PSNR

original = Y.astype(np.float64)
reconstruida = Y_reconstruida.astype(np.float64)

erro = original - reconstruida
MSE = np.mean(erro ** 2)


if MSE == 0:
    PSNR = float("inf")
else:
    PSNR = 10 * np.log10((255 ** 2) / MSE)

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
