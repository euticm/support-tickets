import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime

# --- 1. Funções de Cálculo (Copiadas do seu script VaR) ---
#    (Não é necessário mexer aqui, apenas 'colamos' as funções)
# ==========================================================

def var_historico(returns: pd.Series, alpha: float) -> float:
    """
    VaR Histórico (loss positiva): quantil inferior dos retornos.
    returns: série de retornos (ex.: port_ret).
    alpha: nível de confiança, ex.: 0.99 => P(loss > VaR) = 1 - alpha.
    Retorna um número positivo (perda).
    """
    q = returns.quantile(1 - alpha)  # quantil do lado esquerdo
    return float(max(0.0, -q))

def var_parametrico_normal(mu: float, sigma: float, alpha: float) -> float:
    """
    VaR Paramétrico (Normal): VaR = - (mu + z_alpha * sigma).
    Retorna positivo.
    """
    z = norm.ppf(1 - alpha)  # z < 0 (ex.: alpha=0.99 -> z ~ -2.33)
    var = -(mu + z * sigma)
    return float(max(0.0, var))

def var_mc_normal_univariado(mu: float, sigma: float, alpha: float, n_sims: int = 100_000) -> float:
    sims = np.random.normal(loc=mu, scale=sigma, size=n_sims)
    q = np.quantile(sims, 1 - alpha)
    return float(max(0.0, -q))

def var_mc_normal_multivariado(mu_vec: np.ndarray, cov: np.ndarray, w: np.ndarray,
                                     alpha: float, n_sims: int = 100_000) -> float:
    """
    Simula retornos multivariados normais para os ativos e projeta na carteira.
    """
    sims = np.random.multivariate_normal(mean=mu_vec, cov=cov, size=n_sims)  # (n_sims, n_assets)
    port_sims = sims @ w
    q = np.quantile(port_sims, 1 - alpha)
    return float(max(0.0, -q))

def soma_var_individuais(df_assets, alpha, metodo_col):
    mask = df_assets["Confiança"] == alpha
    return df_assets.loc[mask, metodo_col].sum()  # soma dos VaRs (%) dos ativos

# ==========================================================
# --- 2. Função Principal de Lógica de Cálculo ---
#    (Esta função encapsula todo o processamento do VaR)
# ==========================================================

def run_var_calculation(tickers, weights, start, end, confidence_levels):
    """
    Executa todo o fluxo de cálculo do VaR e retorna os DataFrames e figuras.
    """
    np.random.seed(42)

    # 1) Coleta e retornos log
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    
    # Limpeza de dados
    if prices.empty:
        raise ValueError("Não foi possível baixar os dados. Verifique os tickers ou o período.")
    
    prices = prices.dropna(how="all").dropna(axis=1)
    
    # Se algum ticker falhar, atualiza a lista e os pesos
    valid_tickers = list(prices.columns)
    if len(valid_tickers) < len(tickers):
        st.warning(f"Alguns tickers falharam: {set(tickers) - set(valid_tickers)}. Ajustando pesos.")
        # Re-filtra pesos para tickers válidos
        original_tickers = tickers
        original_weights = weights
        valid_indices = [i for i, ticker in enumerate(original_tickers) if ticker in valid_tickers]
        weights = original_weights[valid_indices]
        tickers = valid_tickers

    if len(weights) == 0:
        raise ValueError("Nenhum ticker válido retornou dados.")

    rets = np.log(prices / prices.shift(1)).dropna()
    mu_vec = rets.mean().values
    cov_mat = rets.cov().values

    # Retorno da carteira
    w = weights / weights.sum()  # Normaliza os pesos para somarem 1
    port_ret = rets @ w
    mu_p = port_ret.mean()
    sigma_p = port_ret.std()

    # 3) VaR por ativo
    rows_assets = []
    for i, t in enumerate(tickers):
        r = rets[t]
        mu_i = r.mean()
        sd_i = r.std()
        for alpha in confidence_levels:
            vh = var_historico(r, alpha)
            vp = var_parametrico_normal(mu_i, sd_i, alpha)
            vmc = var_mc_normal_univariado(mu_i, sd_i, alpha, n_sims=50_000)
            rows_assets.append({
                "Ativo": t,
                "Confiança": alpha,
                "VaR_Histórico_%": 100 * vh,
                "VaR_Param_Norm_%": 100 * vp,
                "VaR_MonteCarlo_%": 100 * vmc
            })
    df_assets = pd.DataFrame(rows_assets)

    # 4) VaR da carteira
    rows_port = []
    for alpha in confidence_levels:
        vh = var_historico(port_ret, alpha)
        vp = var_parametrico_normal(mu_p, sigma_p, alpha)
        vmc = var_mc_normal_multivariado(mu_vec, cov_mat, w, alpha, n_sims=100_000)
        rows_port.append({
            "Confiança": alpha,
            "VaR_Histórico_%": 100 * vh,
            "VaR_Param_Norm_%": 100 * vp,
            "VaR_MonteCarlo_%": 100 * vmc
        })
    df_port = pd.DataFrame(rows_port)

    # 5) Diversificação
    rows_div = []
    for alpha in confidence_levels:
        for metodo_col in ["VaR_Histórico_%", "VaR_Param_Norm_%", "VaR_MonteCarlo_%"]:
            var_sum_indiv = soma_var_individuais(df_assets, alpha, metodo_col)
            var_port = float(df_port.loc[df_port["Confiança"] == alpha, metodo_col])
            ganho_div = var_sum_indiv - var_port
            rows_div.append({
                "Confiança": alpha,
                "Método": metodo_col.replace("_%", "").replace("VaR_",""),
                "Soma_VaRs_Individuais_%": var_sum_indiv,
                "VaR_Carteira_%": var_port,
                "Ganho_Diversificação_%": ganho_div
            })
    df_div = pd.DataFrame(rows_div)

    # 6) Resultados: tabelas (Pivot)
    df_assets_pivot = df_assets.pivot_table(index=["Ativo"], columns="Confiança",
                                            values=["VaR_Histórico_%", "VaR_Param_Norm_%", "VaR_MonteCarlo_%"])
    
    # 7) Gráficos
    figs_assets = []
    for alpha in confidence_levels:
        fig, ax = plt.subplots(figsize=(8, 4))
        m = df_assets[df_assets["Confiança"] == alpha][["Ativo", "VaR_Histórico_%", "VaR_Param_Norm_%", "VaR_MonteCarlo_%"]]
        m.set_index("Ativo").plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"VaR por Ativo — Confiança {int(alpha*100)}%")
        ax.set_ylabel("VaR (% 1 dia)")
        ax.legend(title="Método")
        plt.tight_layout()
        figs_assets.append(fig) # Adiciona a figura à lista
        plt.close(fig) # Fecha para não exibir no console

    # Gráfico da carteira
    fig_port, ax = plt.subplots(figsize=(6, 4))
    df_port.set_index("Confiança")[["VaR_Histórico_%", "VaR_Param_Norm_%", "VaR_MonteCarlo_%"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_title("VaR da Carteira por Método")
    ax.set_ylabel("VaR (% 1 dia)")
    ax.legend(title="Método")
    plt.tight_layout()
    plt.close(fig_port) # Fecha para não exibir no console

    # 8) Escalonar para 10 dias
    days = 10
    scale = np.sqrt(days)
    df_port_10d = df_port.copy()
    for col in ["VaR_Histórico_%", "VaR_Param_Norm_%", "VaR_MonteCarlo_%"]:
        df_port_10d[col] = df_port_10d[col] * scale

    # Retorna todos os objetos que queremos exibir
    return (
        df_port, df_assets_pivot, df_div, df_port_10d,
        figs_assets, fig_port,
        prices.index.min(), prices.index.max(), w
    )


# ==========================================================
# --- 3. Interface do Streamlit (UI) ---
# ==========================================================

# Configuração da página (do seu template)
st.set_page_config(page_title="Calculadora VaR", page_icon="")
st.title("Case Calculadora VaR")
st.write(
    """
    **Calculadora de Value at Risk (VaR)** para ações utilizando dados do **Yahoo Finance**. 
    """
)

st.write(
    """
    Comparando diferentes metodologias, níveis de confiança e o impacto da diversificação.
    """
)
st.write(
    """
    Alunos: *Patrick Viola Montagna e Vinicius Casadei.*
    """
)


# --- 4. Inputs do Usuário (Sidebar) ---
st.sidebar.header("Parâmetros de Entrada")

# Usamos .SA para indicar que queremos adicionar o sufixo para ações BR
add_suffix_sa = st.sidebar.checkbox("Adicionar '.SA' (para ações brasileiras)", value=False)

tickers_input = st.sidebar.text_input(
    "Tickers (separados por vírgula)", "PETR4,VALE3,ITUB4" if add_suffix_sa else "AAPL,MSFT,GOOG"
)
weights_input = st.sidebar.text_input(
    "Pesos (separados por vírgula)", "0.4,0.3,0.3"
)

start_date = st.sidebar.date_input(
    "Data de Início", datetime.date(2020, 1, 1)
)
end_date = st.sidebar.date_input(
    "Data de Fim", datetime.date.today()
)

confidence_levels_input = st.sidebar.multiselect(
    "Níveis de Confiança",
    options=[0.90, 0.95, 0.975, 0.99],
    default=[0.95, 0.975, 0.99]
)

# --- 5. Lógica de Execução e Exibição ---

if st.sidebar.button("Calcular VaR"):

    # 5.1 Parse dos inputs
    try:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if add_suffix_sa:
            tickers = [t + ".SA" if not t.endswith(".SA") else t for t in tickers]
            
        weights_list = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
        
        # 5.2 Validação
        if len(tickers) != len(weights_list):
            st.error(f"Erro: O número de tickers ({len(tickers)}) não é igual ao número de pesos ({len(weights_list)}).")
        elif not tickers or not weights_list:
             st.error("Erro: Preencha os campos de tickers e pesos.")
        elif not confidence_levels_input:
             st.error("Erro: Selecione ao menos um nível de confiança.")
        else:
            weights = np.array(weights_list)
            
            # 5.3 Execução
            with st.spinner(f"Baixando dados e calculando VaR para: {', '.join(tickers)}..."):
                (
                    df_port, df_assets_pivot, df_div, 
                    df_port_10d, figs_assets, fig_port,
                    data_min, data_max, pesos_normalizados
                ) = run_var_calculation(
                    tickers, weights, start_date, end_date, confidence_levels_input
                )
                
                st.success(f"Cálculo concluído! Período dos dados: {data_min.strftime('%Y-%m-%d')} a {data_max.strftime('%Y-%m-%d')}.")
                
                # Exibe os pesos normalizados
                pesos_df = pd.DataFrame({'Ativo': tickers, 'Peso Definido': pesos_normalizados})
                st.write(f"Pesos normalizados da carteira (soma = {pesos_normalizados.sum():.2f}):")
                st.dataframe(pesos_df.style.format({"Peso Definido": "{:.2%}"}), use_container_width=True)

                
                # --- 6. Exibição dos Resultados ---
                st.header("Resultados do VaR (1 dia)")

                st.subheader("VaR da Carteira (% do patrimônio)")
                st.dataframe(df_port.style.format({"Confiança": "{:.1%}", 
                                                    "VaR_Histórico_%": "{:.4f}%",
                                                    "VaR_Param_Norm_%": "{:.4f}%",
                                                    "VaR_MonteCarlo_%": "{:.4f}%"}), use_container_width=True)
                st.pyplot(fig_port)

                st.subheader("VaR por Ativo (% do patrimônio)")
                st.dataframe(df_assets_pivot.style.format("{:.4f}%"), use_container_width=True)
                for fig in figs_assets:
                    st.pyplot(fig)

                st.subheader("Ganho de Diversificação (Soma dos VaRs vs. VaR da Carteira)")
                st.dataframe(df_div.style.format(subset=pd.IndexSlice[:, ['Soma_VaRs_Individuais_%', 'VaR_Carteira_%', 'Ganho_Diversificação_%']], 
                                                  formatter="{:.4f}%")
                                      .format({"Confiança": "{:.1%}"}), 
                                      use_container_width=True)
                
                st.header("Resultados do VaR (10 dias)")
                st.subheader("VaR da Carteira para 10 dias (Regra da Raiz do Tempo)")
                st.dataframe(df_port_10d.style.format({"Confiança": "{:.1%}", 
                                                    "VaR_Histórico_%": "{:.4f}%",
                                                    "VaR_Param_Norm_%": "{:.4f}%",
                                                    "VaR_MonteCarlo_%": "{:.4f}%"}), use_container_width=True)
                
                st.info("Nota: O VaR de 10 dias é escalonado usando a regra da raiz do tempo ($VaR_{10d} = VaR_{1d} \\times \\sqrt{10}$), que assume retornos independentes e identicamente distribuídos (i.i.d.).")

    except Exception as e:
        st.error(f"Ocorreu um erro durante o cálculo:")
        st.exception(e) # Mostra o stack trace completo do erro

else:
    st.info("Preencha os parâmetros na barra lateral e clique em 'Calcular VaR' para iniciar.")