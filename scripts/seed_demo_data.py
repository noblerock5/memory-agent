"""种子数据: user_1 / user_2 的跨天历史对话 + 分层事实

用法: .venv/bin/python scripts/seed_demo_data.py

设计目标: 让"记忆 = 工程实现(数据库检索)"可以被清晰验证——
所有种子内容(生僻姓名/虚构项目细节/具体日期的对话)都是大模型参数知识里不存在的,
模型答对任何一条, 只可能是因为检索注入, 不可能是"它本来就懂"。

时间线(北京时间):
  user_1: 11-06(上周) 自我介绍+一期范围+AI搜索决策 | 11-10 改称呼+TSV超时 | 11-12(昨天) SSH断连+复核决策 | 11-13(今天) 合同流评审
  user_2: 11-05 订单中心重构 | 11-11(前天) OSS直传决策 | 11-12(昨天) 姓名+偏好 | 11-13(今天) 转化率漏斗
"""

import asyncio
from datetime import datetime, timezone

from sqlalchemy import text

from app.core.db import SessionLocal
from app.models.chat_message import ChatMessage
from app.models.memory_fact import MemoryFact
from app.models.user_profile import UserProfile

try:
    from zoneinfo import ZoneInfo

    BJ = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    BJ = timezone.utc


def bj(y: int, mo: int, d: int, hh: int = 0, mi: int = 0, ss: int = 0) -> datetime:
    """北京时间 -> UTC aware (入库统一 UTC)."""
    return datetime(y, mo, d, hh, mi, ss, tzinfo=BJ).astimezone(timezone.utc)


async def main() -> None:
    async with SessionLocal() as s:
        await s.execute(text("TRUNCATE chat_messages, memory_facts, user_profiles RESTART IDENTITY"))
        await s.commit()
        print("已清空全部旧用户数据")

        async def msg(user_id: str, role: str, content: str, dt: datetime) -> ChatMessage:
            m = ChatMessage(user_id=user_id, role=role, content=content, created_at=dt)
            s.add(m)
            await s.flush()
            return m

        # ============ user_1: 池砚舟(→老池) · 克雷登工业 · 证件系统一期 ============
        u1 = "user_1"
        await msg(u1, "user", "我叫池砚舟，在克雷登工业做后端，回复请用极简中文。", bj(2026, 11, 6, 10, 30))
        await msg(u1, "assistant", "记下了，池砚舟。极简中文，收到。", bj(2026, 11, 6, 10, 30, 40))
        m_focus = await msg(u1, "user", "我最近的主力活是证件系统一期的后端。", bj(2026, 11, 6, 10, 31))
        await msg(u1, "assistant", "明白：证件系统一期后端是你的主线。", bj(2026, 11, 6, 10, 31, 20))
        m_vec1 = await msg(u1, "user", "证件系统的 AI 搜索要不要上向量数据库？", bj(2026, 11, 6, 11, 0))
        await msg(u1, "assistant", "一期建议不引入向量库：LLM+Function Calling 直查业务库就够，二期再看规模。", bj(2026, 11, 6, 11, 0, 15))

        m_name = await msg(u1, "user", "以后叫我老池，别叫全名。", bj(2026, 11, 10, 14, 0))
        await msg(u1, "assistant", "好的，老池。", bj(2026, 11, 10, 14, 0, 10))
        m_tsv = await msg(u1, "user", "TSV 大批量导入总在 pg 上超时，怎么破？", bj(2026, 11, 10, 14, 5))
        await msg(u1, "assistant", "经验值：20 并发 + 分批 commit，连接池上限 ≥ 并发数，基本就稳了。", bj(2026, 11, 10, 14, 5, 12))

        m_ssh = await msg(u1, "user", "昨天服务器 SSH 老断连，把我搞烦了。", bj(2026, 11, 12, 9, 20))
        await msg(u1, "assistant", "老问题：批量导入踢空闲连接——分批 commit 之间保持心跳，或配 ClientAliveInterval。", bj(2026, 11, 12, 9, 20, 15))
        m_vec2 = await msg(u1, "user", "再确认一次：一期要不要上向量数据库？", bj(2026, 11, 12, 9, 25))
        await msg(u1, "assistant", "维持 11-06 的结论：一期不上，LLM+FC 直查；二期数据量起来再评估。", bj(2026, 11, 12, 9, 25, 10))

        m_today1 = await msg(u1, "user", "今天同步个进展：证件系统的合同流评审通过了。", bj(2026, 11, 13, 9, 0))
        await msg(u1, "assistant", "好进展，老池——合同流评审通过记下了。", bj(2026, 11, 13, 9, 0, 10))

        # ============ user_2: 苏子衿 · 母婴电商小程序 ============
        u2 = "user_2"
        m2_focus = await msg(u2, "user", "我在做母婴电商小程序，最近要重构订单中心。", bj(2026, 11, 5, 16, 0))
        await msg(u2, "assistant", "订单中心重构是大事——状态机拆分和库存扣减是关键。", bj(2026, 11, 5, 16, 0, 12))

        await msg(u2, "user", "图片存储选 OSS 直传还是后端中转？", bj(2026, 11, 11, 15, 0))
        await msg(u2, "assistant", "建议 OSS 直传：省服务器带宽、少一跳，签名后前端直接 PUT。", bj(2026, 11, 11, 15, 0, 15))
        m2_oss = await msg(u2, "user", "好，那就定 OSS 直传。", bj(2026, 11, 11, 15, 1))
        await msg(u2, "assistant", "定了：图片存储 = OSS 直传。", bj(2026, 11, 11, 15, 1, 8))

        m2_name = await msg(u2, "user", "我叫苏子衿，衿字很难写，备注一下。", bj(2026, 11, 12, 11, 0))
        await msg(u2, "assistant", "记住了，苏子衿（衿 = 衣字旁加今）。", bj(2026, 11, 12, 11, 0, 10))
        m2_style = await msg(u2, "user", "回答我喜欢详细一点，最好带表格对比。", bj(2026, 11, 12, 11, 2))
        await msg(u2, "assistant", "收到：详细 + 表格对比风格。", bj(2026, 11, 12, 11, 2, 8))

        await msg(u2, "user", "今天先看转化率漏斗的数据。", bj(2026, 11, 13, 10, 30))
        await msg(u2, "assistant", "好，转化率漏斗从浏览→加购→下单→支付四段看。", bj(2026, 11, 13, 10, 30, 10))

        # ============ L2 事实(含生命周期/谱系/溯源) + L3 画像 ============
        def fact(user_id, entity, key, value, conf, src=None, vf=None, vt=None, status="active", lc=None):
            return MemoryFact(
                user_id=user_id, entity=entity, key=key, value=value,
                confidence=conf, status=status, source_message_id=src,
                valid_from=vf, valid_to=vt, last_confirmed_at=lc,
            )

        # user_1 —— identity.name 带替代谱系: 池砚舟(superseded) → 老池(active)
        s.add(fact(u1, "user", "identity.name", "池砚舟", 0.95, src=m_name.id,
                   vf=bj(2026, 11, 6, 10, 30), vt=bj(2026, 11, 10, 14, 0), status="superseded", lc=bj(2026, 11, 6, 10, 30)))
        s.add(fact(u1, "user", "identity.name", "老池", 0.95, src=m_name.id,
                   vf=bj(2026, 11, 10, 14, 0), lc=bj(2026, 11, 10, 14, 0)))
        s.add(fact(u1, "user", "work.current_focus", "证件系统一期后端", 0.9, src=m_focus.id,
                   vf=bj(2026, 11, 6, 10, 31), lc=bj(2026, 11, 10, 14, 0)))
        s.add(fact(u1, "user", "preference.like", "极简中文", 0.9,
                   vf=bj(2026, 11, 6, 10, 30), lc=bj(2026, 11, 6, 10, 30)))
        s.add(fact(u1, "user", "goal.milestone", "证件系统合同流评审通过(11-13)", 0.8, src=m_today1.id,
                   vf=bj(2026, 11, 13, 9, 0), lc=bj(2026, 11, 13, 9, 0)))
        s.add(fact(u1, "exp", "exp.solution.TSV导入超时", "TSV大批量导入超时：改20并发+分批提交后稳定", 0.9, src=m_tsv.id,
                   vf=bj(2026, 11, 10, 14, 5), lc=bj(2026, 11, 10, 14, 5)))
        s.add(fact(u1, "exp", "exp.lesson.SSH断连", "长连接被踢：分批commit间保持心跳或配ClientAliveInterval", 0.85, src=m_ssh.id,
                   vf=bj(2026, 11, 12, 9, 20), lc=bj(2026, 11, 12, 9, 20)))
        s.add(fact(u1, "exp", "exp.decision.一期AI搜索", "LLM+FunctionCalling直查业务库，一期不引向量库(11-06定,11-12复核)", 0.9, src=m_vec2.id,
                   vf=bj(2026, 11, 6, 11, 0), lc=bj(2026, 11, 12, 9, 25)))

        # user_2
        s.add(fact(u2, "user", "identity.name", "苏子衿", 0.95, src=m2_name.id,
                   vf=bj(2026, 11, 12, 11, 0), lc=bj(2026, 11, 12, 11, 0)))
        s.add(fact(u2, "user", "work.current_focus", "母婴电商小程序·订单中心重构", 0.9, src=m2_focus.id,
                   vf=bj(2026, 11, 5, 16, 0), lc=bj(2026, 11, 11, 15, 0)))
        s.add(fact(u2, "user", "preference.style", "回答详细+带表格对比", 0.85, src=m2_style.id,
                   vf=bj(2026, 11, 12, 11, 2), lc=bj(2026, 11, 12, 11, 2)))
        s.add(fact(u2, "exp", "exp.decision.图片存储", "OSS直传不走后端中转(省带宽、少一跳)", 0.9, src=m2_oss.id,
                   vf=bj(2026, 11, 11, 15, 1), lc=bj(2026, 11, 11, 15, 1)))

        s.add(UserProfile(user_id=u1, display_name="老池"))
        s.add(UserProfile(user_id=u2, display_name="苏子衿"))
        await s.commit()

        print("种子完成:")
        print("  user_1 (池砚舟→老池 · 证件系统一期): 13 条消息(11-06/11-10/11-12/11-13) + 8 条事实(含谱系)")
        print("  user_2 (苏子衿 · 母婴电商):          10 条消息(11-05/11-11/11-12/11-13) + 4 条事实")


asyncio.run(main())
