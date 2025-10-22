import streamlit as st
import numpy as np
from scipy.optimize import fsolve

# 页面配置
st.set_page_config(
    page_title="悬臂梁一阶固有频率计算器",
    page_icon="📏",
    layout="wide"
)

# 标题与说明
st.title("📏 悬臂梁一阶固有频率计算器")
st.markdown("基于欧拉梁理论，计算矩形截面悬臂梁的一阶固有频率。输入梁的几何参数和材料属性，即可获得精确结果。")


# 定义核心函数
def equation(p, l):
    """悬臂梁特征方程：1 + cos(p*l) * cosh(p*l) = 0"""
    return 1 + np.cos(p * l) * np.cosh(p * l)


def calculate_first_frequency(p, L, E, I, rho, A):
    """计算悬臂梁一阶固有频率"""
    stiffness_mass_ratio = np.sqrt((E * I) / (rho * A))
    frequency = (p ** 2) * stiffness_mass_ratio / (2 * np.pi)
    return frequency


# 输入区域 - 使用两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("几何参数")
    L = st.number_input("梁的长度 (米)",
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


# 计算按钮
if st.button("🔍 计算一阶固有频率", type="primary"):
    try:
        # 单位转换与参数计算
        E = E_GPa * 10 ** 9  # 转换为Pa
        I = (b * h ** 3) / 12  # 截面惯性矩
        A = b * h  # 横截面积

        # 求解特征方程
        initial_guess = 1.875 / L  # 动态初始猜测值
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
        first_frequency = calculate_first_frequency(p_value, L, E, I, rho, A)

        # 结果展示 - 使用卡片式布局
        st.success("计算完成！结果如下：")
        with st.container(border=True):
            result_cols = st.columns(2)
            with result_cols[0]:
                st.info(f"**求解得到的p值**：{p_value:.8f} 1/m")
                st.info(f"**解的误差**：{error:.2e}（越接近0越精确）")
            with result_cols[1]:
                st.info(f"**求解状态**：{'成功' if ier == 1 else '失败'}")
                st.info(f"**一阶固有频率**：{first_frequency:.2f} Hz")

        # 物理意义说明
        st.markdown("""
        ### 结果说明
        - 一阶固有频率是悬臂梁最容易发生共振的频率
        - 误差值小于1e-10时，结果具有工程精度
        - 若求解失败，请检查输入参数是否合理
        """)

    except Exception as e:
        st.error(f"计算出错：{str(e)}")

# 页脚说明
st.markdown("---")
st.caption("基于欧拉梁理论 | 公式：f = (p² / 2π) * √(EI/(ρA)) | 特征方程：1 + cos(pl)cosh(pl) = 0")