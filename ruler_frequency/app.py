import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 中文显示设置
plt.rcParams["font.family"] = ["sans-serif", "Arial Unicode MS", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="尺子一阶固有频率计算器",
    page_icon="📏",
    layout="wide"
)

# 标题与说明
st.title("📏 尺子一阶固有频率计算器")
st.markdown("基于欧拉梁理论，计算矩形截面悬臂梁的一阶固有频率，并绘制**阻尼振动时间曲线**。")


# 定义核心函数
def equation(p, l):
    """悬臂梁特征方程：1 + cos(p*l) * cosh(p*l) = 0"""
    return 1 + np.cos(p * l) * np.cosh(p * l)


def calculate_first_frequency(p, L, E, I, rho, A):
    """计算悬臂梁一阶固有频率（无阻尼固有频率ωₙ）"""
    stiffness_mass_ratio = np.sqrt((E * I) / (rho * A))
    frequency = (p ** 2) * stiffness_mass_ratio / (2 * np.pi)
    return frequency


# 输入区域 - 使用两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("几何参数")
    L = st.number_input("尺子的长度 (米)",
                        min_value=0.01,
                        value=0.3,
                        step=0.01,
                        format="%.3f")

    b = st.number_input("截面宽度 (米)",
                        min_value=0.001,
                        value=0.02,
                        step=0.001,
                        format="%.3f")

    h = st.number_input("截面高度 (米)",
                        min_value=0.0001,
                        value=0.002,
                        step=0.0001,
                        format="%.4f")

with col2:
    st.subheader("材料属性")
    rho = st.number_input("材料密度 (kg/m³)",
                          min_value=100,
                          value=1040,
                          step=10)

    E_GPa = st.number_input("弹性模量 (GPa)",
                            min_value=0.1,
                            value=3.2,
                            step=0.1,
                            format="%.1f")

    # 振动曲线参数设置（新增阻尼系数）
    st.subheader("振动曲线参数")
    duration = st.slider("时间长度 (秒)",
                         min_value=0.5,
                         max_value=5.0,
                         value=2.0,
                         step=0.5)
    amplitude = st.slider("初始振幅 (米)",  # 对应公式中的A₀
                          min_value=0.001,
                          max_value=0.01,
                          value=0.005,
                          step=0.001,
                          format="%.3f")
    # 新增阻尼系数输入（ζ为阻尼比，范围0~1，0.7为临界阻尼附近常用值）
    zeta = st.slider("阻尼比 (ζ)",
                     min_value=0.0,
                     max_value=1.0,
                     value=0.05,
                     step=0.01,
                     help="0=无阻尼，0<ζ<1=欠阻尼，ζ=1=临界阻尼")

# 计算按钮
if st.button("🔍 计算并绘制振动曲线", type="primary"):
    try:
        # 单位转换与参数计算
        E = E_GPa * 10 ** 9  # 转换为Pa
        I = (b * h ** 3) / 12  # 截面惯性矩
        A = b * h  # 横截面积

        # 求解特征方程，得到无阻尼固有频率ωₙ（单位：rad/s）
        initial_guess = 1.875 / L
        solution, infodict, ier, mesg = fsolve(
            equation,
            initial_guess,
            args=(L,),
            xtol=1e-12,
            maxfev=5000,
            full_output=True
        )
        p_value = solution[0]
        error = abs(equation(p_value, L))
        f_n = calculate_first_frequency(p_value, L, E, I, rho, A)  # 固有频率（Hz）
        omega_n = 2 * np.pi * f_n  # 无阻尼圆频率（rad/s）

        # 计算阻尼振动参数（根据公式x(t)=A₀e^(-ζωₙt)cos(ω_d t + φ)）
        omega_d = omega_n * np.sqrt(1 - zeta ** 2) if zeta < 1 else 0  # 阻尼圆频率
        phi = 0  # 初始相位（设为0，可根据需要调整）
        f_nn = omega_d/(2*np.pi)
        # 生成时间序列与位移数据
        t = np.linspace(0, duration, 1000)
        if zeta < 1:  # 欠阻尼状态（有振动）
            displacement = amplitude * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
        else:  # 临界阻尼或过阻尼（无振动，指数衰减）
            displacement = amplitude * np.exp(-zeta * omega_n * t)

        # 结果展示
        st.success("计算完成！结果如下：")
        with st.container(border=True):
            result_cols = st.columns(3)
            with result_cols[0]:
                st.info(f"**p值**：{p_value:.8f} 1/m")
                st.info(f"**无阻尼固有频率**：{f_n:.2f} Hz")
            with result_cols[1]:
                st.info(f"**解的误差**：{error:.2e}")
                st.info(f"**阻尼比**：{zeta:.2f}")
            with result_cols[2]:
                st.info(f"**求解状态**：{'成功' if ier == 1 else '失败'}")
                if zeta < 1:
                    st.info(f"**阻尼修正频率**：{f_nn:.2f} Hz")

        # 绘制阻尼振动曲线
        st.subheader("📈 悬臂梁阻尼振动时间曲线")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 4))

        # 绘制振动曲线
        ax.plot(t, displacement, color='#1E88E5', linewidth=2.5,
                label=f'ζ={zeta:.2f}, fₙ={f_nn:.2f} Hz')

        # 若有阻尼，添加包络线（可视化衰减趋势）
        if zeta > 0 and zeta < 1:
            envelope = amplitude * np.exp(-zeta * omega_n * t)
            ax.plot(t, envelope, 'r--', alpha=0.5, linewidth=1.5, label='衰减包络线')
            ax.plot(t, -envelope, 'r--', alpha=0.5, linewidth=1.5)

        # 图表美化
        ax.set_title(f'Free vibration curve (f = {f_nn:.2f} Hz)', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Displacement (m)', fontsize=12)
        ax.set_xlim(0, duration)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='minor', linestyle='--', alpha=0.3)

        st.pyplot(fig)

        # 物理意义说明
        #st.markdown("""
        #### 结果说明
        #- 公式：$x(t) = A_0 e^{-\zeta\omega_n t} \cos(\omega_d t + \varphi)$
        #- 其中：$\omega_n$=无阻尼圆频率，$\omega_d=\omega_n\sqrt{1-\zeta^2}$=阻尼圆频率，$\zeta$=阻尼比
        #- 当$\zeta=0$时，退化为简谐振动：$x(t)=A_0\cos(\omega_n t + \varphi)$
        #""")

    except Exception as e:
        st.error(f"计算出错：{str(e)}")

# 页脚说明
st.markdown("---")
st.caption("基于欧拉梁理论 | 含阻尼振动方程")