
    prices = download_prices(tickers, start, end)
    daily_returns = prices.pct_change().dropna(how="all").dropna(how="any")
    if daily_returns.empty:
        raise RuntimeError("Could not calculate daily returns.")

    # Annualised arithmetic mean and covariance from daily returns
    mu = daily_returns.mean().to_numpy() * TRADING_DAYS
    cov_df = daily_returns.cov() * TRADING_DAYS
    cov = cov_df.to_numpy()
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)

    gmv = optimise_gmv(mu, cov)
    max_sharpe = optimise_max_sharpe(mu, cov, rf)
    target_weights, target_status = optimise_target_risk(mu, cov, target_vol, gmv)

    stats = pd.DataFrame({
        "Ticker": tickers,
        "Mean daily return (%)": daily_returns.mean().to_numpy() * 100,
        "Annual expected return (%)": mu * 100,
        "Daily standard deviation (%)": daily_returns.std().to_numpy() * 100,
        "Annualised standard deviation (%)": ann_vol.to_numpy() * 100
    })

    gmv_df = weights_table("Global Minimum Variance", gmv, tickers, mu, cov, rf)
    sharpe_df = weights_table("Maximum Sharpe", max_sharpe, tickers, mu, cov, rf)
    combined_weights = [gmv_df, sharpe_df]
    if target_weights is not None:
        combined_weights.append(weights_table("Target Risk", target_weights, tickers, mu, cov, rf))
    all_weights = pd.concat(combined_weights, ignore_index=True)

    # Efficient frontier: solve minimum variance for target returns from GMV return to max asset return.
    frontier_rows = []
    min_ret = portfolio_return(gmv, mu)
    max_ret = float(np.max(mu))
    for target_return in np.linspace(min_ret, max_ret, 60):
        n = len(tickers)
        res = minimize(
            lambda w: portfolio_variance(w, cov),
            gmv.copy(),
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target_return: portfolio_return(w, mu) - t}
            ],
            options={"maxiter": 2500, "ftol": 1e-10}
        )
        if res.success:
            frontier_rows.append({
                "Target/actual return (%)": portfolio_return(res.x, mu) * 100,
                "Annual volatility (%)": portfolio_volatility(res.x, cov) * 100,
                **{f"Weight: {ticker} (%)": res.x[i] * 100 for i, ticker in enumerate(tickers)}
            })
    frontier = pd.DataFrame(frontier_rows)

    dashboard_rows = [
        ["Run date", date.today().isoformat()],
        ["Data start date", start],
        ["Data end date (exclusive)", end],
        ["Assets", ", ".join(tickers)],
        ["Risk-free rate (annual)", rf],
        ["Target annual volatility", target_vol],
        ["Target-risk status", target_status],
        ["GMV expected annual return", portfolio_return(gmv, mu)],
        ["GMV annual volatility", portfolio_volatility(gmv, cov)],
        ["GMV Sharpe ratio", (portfolio_return(gmv, mu) - rf) / portfolio_volatility(gmv, cov)],
        ["Maximum Sharpe expected annual return", portfolio_return(max_sharpe, mu)],
        ["Maximum Sharpe annual volatility", portfolio_volatility(max_sharpe, cov)],
        ["Maximum Sharpe ratio", portfolio_sharpe if False else (portfolio_return(max_sharpe, mu) - rf) / portfolio_volatility(max_sharpe, cov)]
    ]
    if target_weights is not None:
        dashboard_rows += [
            ["Target-risk expected annual return", portfolio_return(target_weights, mu)],
            ["Target-risk realised model volatility", portfolio_volatility(target_weights, cov)]
        ]
    dashboard = pd.DataFrame(dashboard_rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        dashboard.to_excel(writer, sheet_name="Dashboard", index=False)
        prices.rename_axis("Date").to_excel(writer, sheet_name="Prices")
        daily_returns.rename_axis("Date").to_excel(writer, sheet_name="Daily Returns")
        stats.to_excel(writer, sheet_name="Annual Stats", index=False)
        cov_df.to_excel(writer, sheet_name="Covariance")
        all_weights.to_excel(writer, sheet_name="Portfolio Weights", index=False)
        frontier.to_excel(writer, sheet_name="Efficient Frontier", index=False)

    style_workbook(OUTPUT_FILE)
    print(f"\nFinished. Excel output: {OUTPUT_FILE}")
    print(f"Target risk: {target_vol:.2%}; status: {target_status}")
    print("GMV weights:")
    print(pd.Series(gmv, index=tickers).map(lambda x: f"{x:.2%}"))
    print("Maximum Sharpe weights:")
    print(pd.Series(max_sharpe, index=tickers).map(lambda x: f"{x:.2%}"))
    if target_weights is not None:
        print("Target-risk weights:")
        print(pd.Series(target_weights, index=tickers).map(lambda x: f"{x:.2%}"))


if __name__ == "__main__":
    main()
