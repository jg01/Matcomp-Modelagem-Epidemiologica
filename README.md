# Projeto COC351 - Modelo SIR com RK4 para COVID-19 no RJ

Este projeto aplica um modelo epidemiológico SIR aos dados da COVID-19 no estado do Rio de Janeiro.

## Objetivo

Implementar numericamente o sistema de equações diferenciais ordinárias do modelo SIR usando o método de Runge-Kutta de 4ª ordem (RK4), ajustar os parâmetros do modelo por mínimos quadrados e comparar as curvas simuladas com os dados reais.

## Arquivos incluídos

- `sir_rk4_rj.py`: código principal do projeto.
- `RJ_COVID_tratado.csv`: banco tratado com variáveis epidemiológicas.
- `requirements.txt`: bibliotecas necessárias.
- `README.md`: instruções de uso.

## Modelo SIR utilizado

O modelo SIR clássico é dado por:

```text
dS/dt = -beta*S*I/N
dI/dt = beta*S*I/N - gamma*I
dR/dt = gamma*I
```

Onde:

- `S(t)` representa os suscetíveis;
- `I(t)` representa os infectados;
- `R(t)` representa os removidos/recuperados;
- `beta` representa a taxa de transmissão;
- `gamma` representa a taxa de recuperação;
- `N` representa a população total.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o código:

```bash
python sir_rk4_rj.py
```

## Saídas geradas

Após executar o código, serão criados:

- `resultado_sir_rk4_rj.csv`
- `graficos/comparacao_infectados.png`
- `graficos/curvas_sir.png`
- `graficos/erro_ajuste.png`

## Observação metodológica

O ajuste dos parâmetros é feito minimizando o erro quadrático médio entre os infectados observados e os infectados simulados pelo modelo SIR.

Essa lógica segue a ideia de ajuste por mínimos quadrados usada em modelos epidemiológicos: encontrar os parâmetros que fazem a curva simulada se aproximar melhor dos dados reais.
