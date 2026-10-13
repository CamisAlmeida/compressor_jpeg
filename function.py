import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

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

    return blocos, Y, numBlocosAltura, numBlocosLargura, altura, largura
    

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

    return C

# 2.2 DCT 2D

def apply_dct(ycrcb, blockSize=8):
    blocos, Y, numBlocosAltura, numBlocosLargura, altura, largura = divide_blocks(ycrcb)
    C = matrix_dct(blockSize=8)
    print("Matriz da DCT (C):")
    print(C)
    F_blocos = np.zeros_like(blocos, dtype=np.float64)

    # Aplicar a DCT 2D em cada bloco
    for i in range(numBlocosAltura):
        for j in range(numBlocosLargura):
            bloco = blocos[i, j]
            F_blocos[i, j] = C @ bloco @ C.T
    return F_blocos, blocos, Y, numBlocosAltura, numBlocosLargura, altura, largura

# 3 Quantização

def quantization(F_blocos, Q, s, heightBlocks, widthBlocks,):

    Qs = s * Q # Intensidade de quntização

    Fq_blocos = np.zeros_like(F_blocos, dtype=np.int32)

    # Percorre todos os blocos
    for i in range(heightBlocks):
        for j in range(widthBlocks):

            # Coeficientes DCT do bloco
            F = F_blocos[i, j]
            Fq = np.round(F / Qs)
            Fq_blocos[i, j] = Fq.astype(np.int32)
    return Fq_blocos, Qs


# Parte 4  Varredura Zig-Zag
def zigZag_Scan(blockSize, Fq_blocos, heightBlocks, widthBlocks,):
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
        (heightBlocks, widthBlocks, 64),
        dtype=np.int32
    )

    # Percorre todos os blocos quantizados
    for i in range(heightBlocks):
        for j in range(widthBlocks):

            Fq = Fq_blocos[i, j]

            # Reorganiza os 64 coeficientes na ordem zig-zag
            for k, (linha, coluna) in enumerate(ordem_zigzag):
                zigzag_blocos[i, j, k] = Fq[linha, coluna]

    return zigzag_blocos, ordem_zigzag


# Parte 5  Codificação RLE

# 5.1 RLE mínimo sugerido
def rle(heightBlocks, widthBlocks, zigzag_Blocks):
    rle_blocks = []

    for i in range(heightBlocks):
        linha_rle = []

        for j in range(widthBlocks):
            vetor = zigzag_Blocks[i, j]

            run = 0
            cod = []

            for valor in vetor:
                if valor == 0:
                    run += 1
                else:
                    cod.append((run, int(valor)))
                    run = 0

            # Finaliza a codificação do bloco
            cod.append("EOB")

            linha_rle.append(cod)

        rle_blocks.append(linha_rle)
    return rle_blocks


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


# 6 Procedimento inverso

# 6.1 Desquantização

# Parte 6.1 - Decodificação RLE

def decode_rle(rle_blocks, heightBlocks, widthBlocks):
        
    recovered_vector = []

    for i in range(heightBlocks):
        line_vector = []

        for j in range(widthBlocks):
            cod = rle_blocks[i][j]
            vector = []

            for item in cod:
                if item == "EOB":
                    break

                run, value = item

                # Recupera os zeros anteriores ao valor
                vector.extend([0] * run)

                # Recupera o coeficiente não nulo
                vector.append(value)

            # Completa os zeros que faltam até 64 posições
            vector.extend([0] * (64 - len(vector)))

            line_vector.append(vector)

        recovered_vector .append(line_vector)
    return recovered_vector

def reconstruct_Fq(vetores_recuperados, ordem_zigzag,  heightBlocks, widthBlocks):

    Fq_recuperados = np.zeros(
        (heightBlocks, widthBlocks, 8, 8),
        dtype=np.int32
    )

    for i in range(heightBlocks):
        for j in range(widthBlocks):
            vetor = vetores_recuperados[i][j]

            for k, (linha, coluna) in enumerate(ordem_zigzag):
                Fq_recuperados[i, j, linha, coluna] = vetor[k]
    return Fq_recuperados

# 6.2 IDCT matricial

def apply_idct(Fq_recuperados, Q, s, heightBlocks, widthBlocks):

    Qs = s * Q

    F_desquantizados = np.zeros(
        (heightBlocks, widthBlocks, 8, 8),
        dtype=np.float64
    )

    for i in range(heightBlocks):
        for j in range(widthBlocks):
            F_desquantizados[i, j] = Fq_recuperados[i, j] * Qs

    print("Coeficientes desquantizados do primeiro bloco:")
    print(F_desquantizados[0, 0])
    return F_desquantizados


def reconstruct_blocks(F_desquantizados, C, heightBlocks, widthBlocks):
    blocos_reconstruidos = np.zeros(
        (heightBlocks, widthBlocks, 8, 8),
        dtype=np.float64
    )

    for i in range(heightBlocks):
        for j in range(widthBlocks):
            F = F_desquantizados[i, j]

            blocos_reconstruidos[i, j] = C.T @ F @ C
    
    #Desfazer o deslocamento
    blocos_reconstruidos += 128

    return blocos_reconstruidos

def reconstruct_image(blocos_reconstruidos, heightBlocks, widthBlocks, height, width, output):

    blocos_reconstruidos = np.clip(
        np.rint(blocos_reconstruidos),
        0,
        255
    ).astype(np.uint8)

    #Montagem da imagem reconstruída
    height_padded = heightBlocks * 8
    width_padded = widthBlocks * 8

    Y_reconstruida_padded = np.zeros(
        (height_padded, width_padded),
        dtype=np.uint8
    )

    for i in range(heightBlocks):
        for j in range(widthBlocks):
            inicio_linha = i * 8
            inicio_coluna = j * 8

            Y_reconstruida_padded[
                inicio_linha:inicio_linha + 8,
                inicio_coluna:inicio_coluna + 8
            ] = blocos_reconstruidos[i, j]

    # Remover o padding e recuperar o tamanho original
    Y_reconstruida = Y_reconstruida_padded[:height, :width]

    # Criar e salvar a imagem
    imgCinza = Image.fromarray(Y_reconstruida)
    imgCinza.save(output)
    return Y_reconstruida

# 7 Métricas de qualidade
# 7.1 PSNR

def mse_psnr(original, rebuilt ):

    erro = original - rebuilt
    MSE = np.mean(erro ** 2)


    if MSE == 0:
        PSNR = float("inf")
    else:
        PSNR = 10 * np.log10((255 ** 2) / MSE)

    return MSE, PSNR

