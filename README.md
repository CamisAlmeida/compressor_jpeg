# compressor_jpeg

**Disciplina:** 14710 – Arquiteturas Não-Convencionais  
**Período:** 2026.2  
**Projeto:** 1VA – Compressor de Imagens JPEG-like em Python  
**Professor:** Vítor A. Coutinho  
**Instituição:** Universidade Federal Rural de Pernambuco (UFRPE)  
**Departamento:** Departamento de Computação

## Objetivo geral

Implementar, em Python, um compressor de imagens inspirado no padrão JPEG. O projeto integra conversão do espaço de cores RGB para YCrCb, processamento em blocos, aplicação da Transformada Discreta do Cosseno (DCT) 2D, quantização, varredura zig-zag, codificação simples por comprimento de sequências (RLE), cálculo da taxa de compressão em bits por pixel (bpp), reconstrução da imagem e avaliação da qualidade.

Ao final, serão comparadas duas formas de calcular a DCT 2D: uma implementação matricial desenvolvida no projeto e uma implementação rápida e otimizada fornecida por uma biblioteca científica.

## Etapas do projeto

1. **Conversão RGB → YCrCb:** transformação dos canais de cores para obter a luminância Y, utilizada como imagem em tons de cinza.
2. **Deslocamento de nível:** subtração de 128 dos valores de luminância antes da aplicação da DCT.
3. **Divisão em blocos:** particionamento da imagem em blocos de 8 × 8 pixels, com padding quando necessário.
4. **DCT 2D matricial:** construção da matriz ortonormal da DCT-II e aplicação da transformação a cada bloco.
5. **Quantização:** redução da precisão dos coeficientes utilizando uma matriz de quantização e diferentes fatores de escala.
6. **Varredura zig-zag:** reorganização dos 64 coeficientes de cada bloco em um vetor.
7. **Codificação RLE:** representação dos coeficientes quantizados por pares (run, valor), com marcador de fim de bloco (EOB).
8. **Cálculo do bpp:** medição da quantidade média de bits por pixel na representação codificada.
9. **Reconstrução da imagem:** execução das etapas inversas de decodificação RLE, inversão do zig-zag, desquantização, DCT inversa e restauração dos níveis de intensidade.
10. **Avaliação da qualidade:** cálculo das métricas PSNR e SSIM e geração dos gráficos PSNR × bpp e SSIM × bpp.
11. **Comparação das implementações da DCT:** validação numérica e comparação do tempo de execução entre a versão matricial e a versão otimizada da biblioteca SciPy.
12. **Análise de desempenho:** cálculo do speedup e discussão dos resultados obtidos.

## Tecnologias e bibliotecas

- Python
- NumPy
- SciPy
- Matplotlib
- Pillow ou OpenCV para leitura, exibição e salvamento de imagens
- scikit-image para o cálculo do SSIM

## Resultados esperados

- Código-fonte completo do compressor JPEG-like.
- Implementação matricial da DCT 2D e sua inversa.
- Segunda versão do codec utilizando uma DCT otimizada de biblioteca.
- Imagem original e imagens reconstruídas para diferentes fatores de quantização.
- Tabelas com bpp, percentual de coeficientes zero, PSNR, SSIM, tempos de execução e speedup.
- Gráficos de qualidade em função da taxa de compressão.
- Validação numérica das duas implementações da DCT.
- Relatório com metodologia, resultados e discussão do desempenho.

**Objetivo final:** construir um codec funcional e relacionar transformação, quantização, taxa de compressão, qualidade da imagem reconstruída e custo computacional.
