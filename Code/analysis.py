"""
Esse código deve gerar toda a analise dos dados gerados pelo backtest, deve ter como input o id do dataset que ele vai analiser e deve crair dentro da pasta Analysis, uma sib dir com o nome do id que vamos analisar,ela olha os dados processados do id e gera varios graficos, quero que ela tenha uma subpasta que gera um grafico pra predicao de cada dia que o modelo preveu, com a curva do modelo e a curva do mercado no futuro(tambem quero uma ultima curva ou scatter points, algo do tipo, que mostre qual era o preço de cada um dos tenores naquele dia, ou seja no dia que foi feita a predição, o mercado tambem tinha a sua "predicao do futuro", que é o preco dos futures naquele dia, quero compara isso com o preco futuro do mercado e o nosso modelo), no passado mostra apenas os dados reais, na pasta vai ter um arquivo .png com a imagem do grafico par cada dia.

tambem gere outra pasta com o feedback final de backtest do modelo, um grafico do valor da nossa carteira e compara com a taxa tipo cid, que nao sei qual o nome, dos estados unidos, e compare com a estrategia de buy and hold que seria o 'jeito mais burro de investir' mas que acompanha o desempenho do mercado e é um bom parametro pra comparar o desempenho do modelo com o vies do mercado,. 

dentro dessa segunda grande pasta com essa imagem do desempenho, quero que voce coloque o valor dos parametros do modelo em cada dia em um grafico, junto com o valor do mercado, assim da pra ver como o modelo se comportou em cada dia e como ele se comportou em relação ao mercado

E também crie uma nova grande pasta(agora sao 3), com a comparacao por tenores, dentro dela vai ter uma pasta por tenor e pra cada um, quero ver um grafico falando dessa comparacao geral de lucro mas só pra esse tenor, tambem quero um grafico com a curva do mercado  e nossas decisoes de compra e venda, em cima do grafico, com as nossas decisoes a amostra. NEsses graficos das decisoes de venda quero que tenha um grafico na mesma imagem, a baixo desse e no eixo x o tempo e no eixo y uma relacao do valor do nosso modelo comparado com o valor que o mercado está precificando aquele future. formula: (nosso preco - preco mercado) / preco mercado, entao ele vai ficar negativo se o mercado esta precificando demais e positivo se estiver subvalorizado.


Se tem alguma analise que deve ser fefita e está faltando crie uma nova pasta e chame de outros e coloque tudo lá.

Se eu esqueci de pedir algum dado(util para essa analise) na geracao do backtest, pode colocar ele lá para poder chamar aqui.

"""