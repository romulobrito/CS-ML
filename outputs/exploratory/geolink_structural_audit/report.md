# Auditoria estrutural do piloto GeoLink

Fold 0 do GeoLink, protocolo congelado do piloto. Não é o experimento F3 e não é o poço 861. Os canais não têm nome petrofísico neste arquivo: o alvo é o canal 0; os auxiliares são os canais 1 e 2; o canal 3 não entra no piloto. O metadado de poço só registra a contagem de janelas.

## P1. O experimento original foi reproduzido?

Sim, no critério pedido: a média de três sementes do RMSE nas 51 posições não observadas fica a menos de 0,002 dos valores com quatro casas. A maior distância é 0,00080, no CSGM livre com máscara aleatória (0,49420 contra 0,4950).

O que bate de forma exata, ou praticamente exata:

- Poços de treino, validação e teste, com 56, 55 e 72 janelas.
- Lambda escolhido na validação: 0,01 em cinco células e 0,1 na lacuna da semente 23.
- Interpolação linear e Ridge auxiliar, em todas as sementes, com diferença da ordem de 1e-16.
- Sementes 23 e 41 nos métodos neurais, com diferença máxima de cerca de 3e-5.

A semente 7 não reproduz o CSV empacotado no mesmo grau. A maior distância é 0,00235, no CSGM livre com máscara aleatória. O prior da mesma semente, que não passa pela otimização latente, sai 1,20824 aqui e 1,20756 no pacote. Executar a função `fit_ae` do script congelado nesta máquina dá o mesmo 1,20824. A diferença não veio da exportação, da máscara, do ruído nem do split. O treino da semente 7 neste PyTorch 2.6.0+cu124 é determinístico, mas não é bit a bit o treino que gerou o CSV empacotado. As médias de três sementes absorvem esse deslocamento e continuam dentro de 0,002.

As predições usadas abaixo são as deste processo. Não houve segundo treino.

## P2. O CLP tem RMSE menor que a interpolação em algum poço, nas três sementes?

Não. Nas 12 poços, a média entre sementes do RMSE do CLP menos o da interpolação é positiva nas duas máscaras e nas três regiões. O intervalo bootstrap por poço, 5000 reamostras, semente 2026, fica inteiro acima de zero.

Na região de todas as posições não observadas:

| Máscara | Média CLP - interpolação | Intervalo 2,5-97,5% | Poços em que o CLP ganha nas 3 sementes |
| --- | ---: | --- | ---: |
| Aleatória | +0,159 | 0,116 a 0,215 | 0/12 |
| Lacuna | +0,231 | 0,166 a 0,327 | 0/12 |

Há um único caso isolado: poço `35_11-6`, lacuna, semente 7, diferença -0,070 no RMSE agregado do poço. A média desse poço entre as três sementes continua +0,191. Não é uma vantagem consistente. Essa semente é também a que não reproduz o CSV empacotado bit a bit.

## P3. O CLP tem menor RMSE das diferenças consecutivas que a interpolação?

Não, em nenhuma região agregada. Uma diferença só entra na conta se os dois extremos são não observados, estão na região e não cruzam a fronteira 15|16 nem 47|48.

Média das três sementes, região completa:

| Máscara | CLP | CSGM livre | Interpolação |
| --- | ---: | ---: | ---: |
| Aleatória | 0,477 | 0,521 | 0,295 |
| Lacuna | 0,465 | 0,544 | 0,289 |

O CLP perde para a interpolação em 12/12 poços na máscara aleatória, nas três regiões, nas três sementes. Na lacuna, perde em 12/12 na região completa e no centro. Fora da lacuna há uma única semente, no poço `35_11-10`, com diferença -0,021; a média do poço continua positiva.

## P4. Quando o CLP oscila mais, essas oscilações acompanham o sinal?

Não há evidência disso. A razão de amplitudes das diferenças, desvio reconstruído sobre desvio verdadeiro, fica em torno de 1,24 no CLP, 1,43 a 1,58 no CSGM livre e 0,18 a 0,22 na interpolação, na região completa. O CLP varia demais; a interpolação varia de menos; nenhum dos dois fica perto de 1 no centro da lacuna, onde a interpolação cai para cerca de 0,07.

A correlação das diferenças com o sinal verdadeiro, reunida pelos pares e não pela média das correlações de cada janela, é cerca de -0,01 a 0,00 no CLP, perto de zero no CSGM livre, e 0,12 a 0,16 na interpolação. CLP menos interpolação nessa correlação tem média -0,171 na máscara aleatória e -0,127 na lacuna. Os dois intervalos ficam inteiros abaixo de zero, e nenhum poço dá a vitória ao CLP nas três sementes.

Oscilar mais que a interpolação, com correlação nula, não é recuperação da variação verdadeira.

## P5. As vantagens aparecem em vários poços ou só em exemplos?

A desvantagem do CLP frente a interpolação, no RMSE e no RMSE das diferenças, aparece nos 12 poços. Não depende da janela desenhada primeiro.

A redução de oscilação do CLP frente ao CSGM livre aparece em 11 ou 12 poços e o intervalo da média não cruza zero. Isso é estável, mas é uma comparação entre dois métodos que ambos erram a forma do sinal.

Os gráficos usam a primeira janela exportada de cada semente, que é a linha 0 do arranjo, poço `15_9-23`, índice de origem 128, e também a primeira janela de cada um dos 12 poços na semente 7. Essa escolha foi fixada pela ordem do split, não pelo erro.

## P6. O condicionamento melhora o CSGM livre?

No RMSE convencional, não de forma estável. CLP menos CSGM, região completa: média +0,001 na máscara aleatória e +0,024 na lacuna. Os dois intervalos cruzam zero. Em cinco poços da máscara aleatória o CLP ganha nas três sementes; na lacuna, em um poço. A média dos poços não se separa de zero.

No RMSE das diferenças, o CLP fica abaixo do CSGM livre: média -0,041 na máscara aleatória e -0,076 na lacuna, intervalos inteiros negativos. A correlação dessas diferenças com o sinal não acompanha a queda: CLP menos CSGM fica em cerca de +0,003 a +0,004 e o intervalo cruza zero.

O condicionamento, neste piloto, reduz variação espúria do CSGM livre. Não mostra que essa variação a menos esteja alinhada ao sinal, nem que o RMSE melhore de modo reproduzível entre poços.

## P7. O que fica negativo ou inconclusivo?

Negativo para a hipótese de que o CLP recupera melhor as variações locais apesar do RMSE pior:

- O RMSE das diferenças também é pior que o da interpolação.
- A correlação das diferenças é pior que a da interpolação.
- A amplitude do CLP passa de 1 e a correlação fica em torno de zero.

Inconclusivo, e não uma vitória do condicionamento:

- RMSE do CLP contra o CSGM livre, porque o intervalo por poço cruza zero.
- Qualquer leitura petrofísica dos canais, porque o arquivo não traz os nomes.

Não se estende este fold do GeoLink ao F3 nem ao 861. Há um único split, 56 janelas de treino, 12 poços de teste com seis janelas cada, e três sementes. As 72 janelas não são 72 poços. As três sementes não são 36 poços.

## Protocolo que foi mantido

Treino 56 janelas de 56 poços, validação 55 janelas de outros 10, teste 72 janelas de outros 12. O teste sai do arquivo `geolink_fold_0_well_log_sliced_val.npy`; os poços não se misturam com o treino. Janela original 256, passo 4, comprimento 64. Treze observações. Máscara aleatória, ou 6 posições em 0-15 e 7 em 48-63. Ruído gaussiano 0,01. Autoencoder 64-64-16-64-64, ReLU, 180 épocas, Adam 0,001, weight decay 1e-5, como no script congelado. Inferência com 220 passos, duas reinicializações, passo 0,05. Lambda só na validação, na grade de 1e-6 a 1e-1. Ridge latente com alpha 1. Ridge auxiliar direto com alpha 10. Interpolação nos índices 0..63, com valor extremo constante. O mesmo decodificador serve ao CLP e ao CSGM livre dentro da semente.

O zip pedido como `public-processed-datasets(1).zip` não estava em Downloads. Os arrays do fold 0 já extraídos em `/home/romulo/Downloads/public-processed-datasets` têm o mesmo CRC do `public-processed-datasets.zip`. O zip `geolink_continuacao_auditoria.zip` não foi encontrado e não foi substituído por outro.

Nada foi commitado, nada foi enviado, e a `main` não foi alterada.
