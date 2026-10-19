"""Embed the inspected architecture PNG without altering frozen result builders.

Source topology: tests/spin_graph_20260920/model.py and this package's
common.py/run.py. Raster authored with built-in image generation (backend
identity unspecified), ccf-visual-composer formal-roman layout. No source data
or measured plots were sent to generation. Current PNG is not vector-editable.
Figure brief: three aligned regions, input/Gram features, scalar-vector
residual graph blocks with parallel temporal attention, training vs spherical
flow sampling. Correct equations are additionally typeset in the HTML below.
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
START = '<!-- p2-architecture:start -->'
END = '<!-- p2-architecture:end -->'


def embed(document):
    image = HERE/'assets/model_architecture.png'
    if not image.exists():
        return document
    if START in document:
        left, rest = document.split(START, 1)
        _, right = rest.split(END, 1)
        document = left+right
    encoded = base64.b64encode(image.read_bytes()).decode('ascii')
    section = START+'''<section id="p2-model-architecture">
<h2>当前实际模型架构：局域标量—向量图网络 + 条件球面流</h2>
<p>这张图对应本轮三个 seed 使用的同一架构：宽度64、3个残差块、4头时间注意力、93,972个可训练参数。展示的是已实现的局域模型，不包含后续计划中的多尺度层。</p>
<figure><img style="width:100%;height:auto" alt="P2模型架构：输入及Gram不变量、并行空间消息与时间注意力、球面流训练和生成" src="data:image/png;base64,'''+encoded+'''">
<figcaption>由实际代码核对的示意图；磁链箭头是结构示意，不是模拟结果。A看输入，B看网络内部，C比较训练与实际生成。可放大网页查看，图片已内嵌，离线无需另外复制PNG。</figcaption></figure>
<h3>A：网络接收什么</h3>
<p>X<sub>τ</sub>[k]是当前运输步骤中的第k个物理帧；相邻帧也来自同一份当前路径草稿，不是额外提供的真实未来。7个向量为当前磁矩、给定初态、前后草稿帧和3个晶体基向量；49个点积不变量加上τ、物理时间t、标准化温度，共52个输入标量。端点帧的邻帧用边缘重复处理。</p>
<p>尺寸决定开放链图的节点数与邻接关系；边界固定绑定模型。温度经训练数据常量处理后为零，本轮并未学习跨温度变化。seed控制球面噪声抽样，不是神经网络物理特征。网络没有读取J、K或Hamiltonian能量；晶体基与磁矩需联合旋转。</p>
<h3>B：空间、时间与等变性怎样结合</h3>
<p>A的Linear输出初始化标量特征h，当前草稿磁矩初始化向量特征v。归一化后的标量特征同时进入空间图消息和逐格点时间注意力，两路相加后经FiLM及残差更新；不是把三维磁矩坐标直接交给普通注意力混合。每个块使用独立参数。</p>
<p>向量消息为 <code>a<sub>ij</sub> v<sub>j</sub> + b<sub>ij</sub>(v<sub>i</sub> × v<sub>j</sub>)</code>，a、b由标量消息网络产生。邻居均值乘0.1后残差更新v；标量支路也使用0.1残差尺度。三块之后才使用输出头：14个标量分为7个直接项系数和7个叉积项系数，对向量基加权，再做切向投影。输出基的当前磁矩槽替换为最终v，其余基保留。</p>
<p>切向投影为 <code>P<sub>x</sub>(u)=u−(x·u)x</code>，首帧速度置零。它与SO(3)协变操作、球面更新共同构成结构约束；这些约束本身不证明随机路径统计准确。</p>
<h3>C：训练和生成不是同一条过程</h3>
<p>训练使用真实路径Y和独立均匀球面噪声Z，沿测地插值得到X<sub>τ</sub>及目标速度U<sub>τ</sub>，最小化未来帧的流匹配误差。本轮没有能量辅助损失。生成不读取真实未来：从Z出发，反复预测整段路径速度并做球面Exp更新，每次重新固定首帧为本次给定S₀。</p>
<p>循环中始终是当前草稿路径，只有τ=1时作为最终输出。16/32/64/128是流运输积分步数；物理输出均为1 ps、11帧、0.1 ps间隔。开放链L16/L25/L32均属于本轮训练覆盖尺寸。初态来自给定配置或已登记准备协议，不使用学习式初态生成器。</p>
<p>拓扑依据：<code>tests/spin_graph_20260920/model.py</code>、<code>tests/bauer_p2_20260923/common.py</code>、<code>run.py</code>。绘制使用ccf-visual-composer与内置图像工具，最终产物为PNG；说明与公式是网页可选取文本。</p>
</section>'''+END
    title = '<h1>P0–P2：Bauer 局域图模型首轮闭环</h1>'
    if title not in document:
        raise ValueError('P2 page anchor missing; refusing to insert into unrelated HTML')
    return document.replace(title, title+section, 1)


def main():
    for name in ('index.html', 'fragment.html'):
        path = HERE/'assets'/name
        original = path.read_text(encoding='utf-8')
        updated = embed(original)
        assert updated.count(START) == 1 and updated.count(END) == 1
        assert '\ufffd' not in updated
        if updated != original:
            path.write_text(updated, encoding='utf-8')
    from tests.progress_dashboard.build import main as dashboard
    dashboard()


if __name__ == '__main__':
    main()
