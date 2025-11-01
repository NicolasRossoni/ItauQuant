"""
Aqui é o fluco principal do código, praticanebte sempre rodaremos aqui

input : { # Devem ser strings definidas no inicio do código, do lado de cada string deve ter "# uma explicacao de o que ela faz"

dataset que será usado, # O nome/id da pasta do raw que vamos usar (nome que criamos no arquivo de dowload, o modelo apenas usara os dados de lá aqui)
start date de dados passados, # O modelo vai ser calibrado para o seu primeiro dia de testes com dados que comecao nessa data
start date de testes, # O modelo vai comecar a testar a partir dessa data(só com os dados dos dias que já passaram para calibrar o parametros)
end date de testes,  # O modelo vai testar todos os dias até chegar aqui


}

output : {

# Fale que tem um print minimalista no terminal para mostrar o status do código
# e guarda os arquivos na pasta data/processed/{id_especificado_nos_inputs # mesmo id do nome do dataset escolhido}. 

}

O código vai testar o modelo pra cada dia e testar o training stretegy também(que também usa a previsao do modelo). Para cada dia do teste ele vai crair uma pasta, dentro da pasta designada para o outpus do código, com a data do dia no nome da sub pasta. Dentro desta subpasta deve ter todos os logs do dias, em quantos arquivos forem necessarios, os dados aqui vao ser usados para a analise e ela deve ter acesso a tudo que o código fez nao só variaveis finais de lucro e etc, então eu quero que voce coloque tudo que puder ser util pra qualquere tipo de analise aqui(como demora muito pra rodar o código nao quero que fique faltando nada aqui pra na hora da analise eu descobrir que preciso rodar o código de novo). Mas as principais que eu quero é uma subpasta com as predicoes do modelo futuras e com os dados futuros do modelo(com todos os dados anteriores ao inicio daquela data também, a ideia é eu poder gerear um grafico para cada dia com a predicao do modelo, mas isso será na analise, estou explicando só pra vc entender a nocao gral do projeto) gostaria que tivesse o vetor de decisao do modelo de treinamento naquele dia também, E agora voltando para o diretorio geral de output, a gente vai ter essaas pastas com dados de cada dia e vai ter uma subpasta principal que vai ter o log do desempenho geral do método, e como nas outras pastas, quero que tenha todos os tipos da dados aqui para poder fazer a analise, como por exemplo o valor da nossa carteira em cada dia, quais foram os comandos de venda e compra e hold de cada dia e etc, deixe tudo bem documentado nos comentarios do código pra nao ser dificil mexer com a estrutura depois na analise

Os arquivos que sao guardados aqui 

# O código deve ser simples e só ter a logica geral de chmar pouquissimas funcoes que da pra entender como os dados estão sendo manipulados(quase como seria num fluxograma), a sintax de verdade está nos arquivos no src/ que deve ter suas funções importadas aqui
"""