    # ================================================================
    # 2. 趋势图（近24小时）—— 已修复标签错位问题
    # ================================================================
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    st.caption("🟦 实线 = 实测/进水 | 虚线 = 预测 | 各指标颜色区分")

    recent_data = st.session_state.data_buffer.get_recent(24)
    if len(recent_data) > 1:
        # ----- 构建包含全部5个指标的数据框 -----
        df_trend = pd.DataFrame([{
            'timestamp': d['timestamp'],
            'inlet_COD': d['inlet']['COD'],
            'inlet_NH3': d['inlet']['NH3-N'],
            'inlet_TP': d['inlet']['TP'],
            'inlet_TN': d['inlet']['TN'],
            'inlet_SS': d['inlet']['SS'],
            'outlet_COD_real': d['outlet']['COD'] if d['outlet'] else None,
            'outlet_COD_pred': d['pred_outlet']['COD'] if d['pred_outlet'] else None,
            'outlet_NH3_real': d['outlet']['NH3-N'] if d['outlet'] else None,
            'outlet_NH3_pred': d['pred_outlet']['NH3-N'] if d['pred_outlet'] else None,
            'outlet_TP_real': d['outlet']['TP'] if d['outlet'] else None,
            'outlet_TP_pred': d['pred_outlet']['TP'] if d['pred_outlet'] else None,
            'outlet_TN_real': d['outlet']['TN'] if d['outlet'] else None,
            'outlet_TN_pred': d['pred_outlet']['TN'] if d['pred_outlet'] else None,
            'outlet_SS_real': d['outlet']['SS'] if d['outlet'] else None,
            'outlet_SS_pred': d['pred_outlet']['SS'] if d['pred_outlet'] else None,
        } for d in recent_data])

        # 定义5个子图标题
        subplot_titles = ('COD', 'NH₃-N', 'TP', 'TN', 'SS')
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=subplot_titles)

        # 定义每个指标的配置 (列名前缀, 进水颜色, 实测颜色, 预测颜色, 限值)
        indicators_config = [
            ('COD', '#E74C3C', '#2E86AB', '#2E86AB', DESIGN_LIMITS['COD']['value']),
            ('NH3', '#F39C12', '#27AE60', '#27AE60', DESIGN_LIMITS['NH3-N']['value']),
            ('TP', '#8E44AD', '#F39C12', '#F39C12', DESIGN_LIMITS['TP']['value']),
            ('TN', '#1F77B4', '#FF7F0E', '#FF7F0E', DESIGN_LIMITS['TN']['value']),
            ('SS', '#9467BD', '#D62728', '#D62728', DESIGN_LIMITS['SS']['value'])
        ]

        for row, (prefix, in_color, real_color, pred_color, limit) in enumerate(indicators_config, start=1):
            # 进水
            fig.add_trace(
                go.Scatter(x=df_trend['timestamp'], y=df_trend[f'inlet_{prefix}'],
                           name=f'进水 {prefix}', line=dict(color=in_color, width=2)),
                row=row, col=1
            )
            # 实测出水（如果有）
            mask_real = df_trend[f'outlet_{prefix}_real'].notna()
            if mask_real.any():
                fig.add_trace(
                    go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real][f'outlet_{prefix}_real'],
                               name=f'出水 {prefix} 实测', line=dict(color=real_color, width=2.5)),
                    row=row, col=1
                )
            # 预测出水
            mask_pred = df_trend[f'outlet_{prefix}_pred'].notna()
            if mask_pred.any():
                fig.add_trace(
                    go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred][f'outlet_{prefix}_pred'],
                               name=f'出水 {prefix} 预测', line=dict(color=pred_color, width=2, dash='dot')),
                    row=row, col=1
                )
            # 限值线
            fig.add_hline(y=limit, line_dash="dash", line_color="red", row=row, col=1)

        # ----- 为每条曲线添加末端标签（修正：x轴偏移2分钟，y轴动态偏移） -----
        # 按索引遍历所有trace，为每个trace的最后一个非空点添加注释
        # 为了错开，每个trace的x偏移为 (idx+1)*2 分钟
        for idx, trace in enumerate(fig.data):
            # 获取非空的x,y值
            x_vals = trace.x
            y_vals = trace.y
            # 找到最后一个非空点（排除None）
            valid_indices = [i for i, (x, y) in enumerate(zip(x_vals, y_vals)) if x is not None and y is not None]
            if not valid_indices:
                continue
            last_idx = valid_indices[-1]
            x_last = x_vals[last_idx]
            y_last = y_vals[last_idx]
            # 偏移量：每个trace偏移 (idx+1)*2 分钟，避免重叠
            offset_minutes = (idx + 1) * 2
            x_offset = x_last + timedelta(minutes=offset_minutes)
            # 根据y值决定上下偏移，若y值较大则向上，否则向下
            # 获取该子图的y范围（但这里简单处理：根据y_last相对于该行限值的比例）
            # 简单策略：y值越大的线向上偏移，否则向下
            # 取该trace的所属行（但这里无法直接获取，我们可以通过name判断）
            # 更简单：根据trace的索引奇偶交替
            yshift = 15 if (idx % 2 == 0) else -15
            # 如果y_last很大（比如进水），可以适当增加偏移
            # 根据trace name是否包含"进水"来调整
            if '进水' in trace.name:
                yshift = 20 if (idx % 2 == 0) else -20
            else:
                yshift = 12 if (idx % 2 == 0) else -12

            fig.add_annotation(
                x=x_offset,
                y=y_last,
                text=trace.name,
                showarrow=False,
                yshift=yshift,
                font=dict(size=9, color=trace.line.color),
                xanchor='left'
            )

        fig.update_layout(height=650, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="时间（北京时间）", row=5, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 数据收集中... 请等待更多数据点（至少2个时间点）")
