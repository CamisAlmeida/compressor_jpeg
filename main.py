import numpy as np
from PIL import Image
import math

# Entrada da imagem 
INPUT_FILE = 'foto1.jpeg'
OUTPUT_GRAY_FILE = 'foto1Cinza.png' # arquivo de saida da imagem cinza
OUTPUT_FILE = 'foto1Processada.png' # arquivo de saida da imagem procesada

# 1 Requisitos de entrada e preparação

# 1.1 conversão RGB → YCrCb 

# Abrir a imagem
img = Image.open(INPUT_FILE).convert('RGB')

# Converter a imagem em uma matriz NumPy
rgb = np.array(img, dtype=np.float64)

# Verificar as dimensões
print("Dimensões da imagem:", rgb.shape)
print("Tipo dos dados:", rgb.dtype)
print("Primeiro pixel de rgb:", rgb[0, 0])

matrizBase = np.array([
    [ 0.299000,  0.587000,  0.114000],
    [ 0.500000, -0.418688, -0.081312],
    [-0.168736, -0.331264,  0.500000]
], dtype=np.float64)

deslocamento = np.array([0, 128, 128], dtype=np.float64)

# YCrCb 
ycrcb = rgb @ matrizBase.T + deslocamento


print("Dimensões da imagem:", ycrcb.shape)
print("Tipo dos dados:", ycrcb.dtype)
print("Primeiro pixel de YCrCb:", ycrcb[0, 0])


# Salvar a imagem em tons de cinza

# Arredondar os valores e converter para 8 bits
imagemCinza = np.clip(np.round(ycrcb[:, :, 0]), 0, 255).astype(np.uint8)

# Criar e salvar a imagem
imgCinza = Image.fromarray(imagemCinza)
imgCinza.save(OUTPUT_GRAY_FILE)

print("Imagem em tons de cinza salva:", OUTPUT_GRAY_FILE)

# Parte 1.2 - Deslocamento de nível

# Extrair o canal Y (luminância)
Y = ycrcb[:, :, 0]

#Subtrair 128 do canal Y
imagemDeslocada = Y - 128

print("primeiro pixel da imagem deslocada:", imagemDeslocada[0, 0])


# Parte 1.3 - Divisão em blocos 8x8

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

# 2 Versão A - DCT 2D pela operação matricial

# 2.1 Construção da matriz da DCT


# Parte 2.1 - Construção da matriz da DCT

N = 8 # tamanho do bloco

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

# 2.2 DCT 2D

F_blocos = np.zeros_like(blocos, dtype=np.float64)

# Aplicar a DCT 2D em cada bloco
for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):
        bloco = blocos[i, j]
        F_blocos[i, j] = C @ bloco @ C.T


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

# Parte 3 - Quantização

# Fator de escala
s = [0.5, 1, 2, 4]

# Intensidade de quntização
Qs = s[1] * Q

# Armazena os coeficientes quantizados
Fq_blocos = np.zeros_like(F_blocos, dtype=np.int32)

# Percorre todos os blocos
for i in range(numBlocosAltura):
    for j in range(numBlocosLargura):

        # Coeficientes DCT do bloco
        F = F_blocos[i, j]

        Fq = np.round(F / Qs)

        # Armazena os coeficientes quantizados
        Fq_blocos[i, j] = Fq.astype(np.int32)

print("Tabela Q:")
print(Q)

print("Tabela Qs:")
print(Qs)

print("Primeiro bloco quantizado:")
print(Fq_blocos[0, 0])
