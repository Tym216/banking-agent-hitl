# AGENTS.md

## 架构红线（不可违反）
- 权限裁决与审批闸门在框架层执行，不依赖 LLM 输出
- 银行核心数据只能经 MCP 工具访问；agent 代码禁止直接读写 bank 库
- 两库分离：agent.db（会话/用户/审计）与 bank_core.db（账户/交易/工单）
- 优先考虑配置驱动，但不要复杂化配置

## 数据规范
- 写操作必须事务化，数值更新原子化
- 有副作用的工具必须有幂等键（重放不得产生第二次效果）
- 全链路动作留痕，可回放
- 数据库连接默认 `PRAGMA foreign_keys = ON`；thread_id 系外键要求先建会话再落消息
- schema 含 SQLite 专属语法（`datetime('now')`、AUTOINCREMENT、PRAGMA），迁移 PostgreSQL 时必须替换
- 登录/角色变更等敏感操作写入 append-only 的 audit_logs，不得 UPDATE/DELETE

## 工作约定
- 大任务先讨论方案，拆 phase，定位原代码文件，并列出执行清单，经用户确认后再动手
- 改接口时同步检查所有调用点（包括测试）

## Git
- 英文 Conventional Commits；README 中英互链
