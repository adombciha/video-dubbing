"""Traditional Chinese UI for the local dubbing workflow."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from dubbing import (
    DIRECTIONS,
    DubbingError,
    Settings,
    SpeechTimingError,
    apply_edits,
    check_environment,
    prepare_video,
    render_job,
    translate_segments,
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: auto; }
button { min-height: 44px; }
input, textarea { font-size: 16px !important; }
@media (max-width: 600px) {
    .gradio-container { padding: 12px !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""
PROGRESS = gr.Progress()


def table_rows(job):
    return [[str(s.id), f"{s.start:.2f}–{s.end:.2f} 秒", s.source, s.target] for s in job.segments]


def build_app(settings=None):
    settings = settings or Settings.from_env()

    def environment_status():
        return "\n".join(
            f"{'[OK]' if ok else '[待設定]'} {name}：{detail}"
            for name, ok, detail in check_environment(settings)
        )

    def prepare(video, direction, progress=PROGRESS):
        yield None, [], "辨識與翻譯中，請稍候。", None, None, None, None
        try:
            job = prepare_video(video, direction, settings, progress)
            yield job, table_rows(job), "辨識與翻譯完成。請校對產品名稱、數字及譯文，再產生配音。", None, None, None, None
        except DubbingError as exc:
            yield None, [], f"處理失敗：{exc}", None, None, None, None
            raise gr.Error(str(exc)) from exc

    def retranslate(job, rows, progress=PROGRESS):
        if job is None:
            raise gr.Error("請先上傳影片並完成辨識。")
        try:
            apply_edits(job, rows)
            translate_segments(job.segments, job.target_language, settings, progress)
            job.save()
            return job, table_rows(job), "已依修正後的原文重新翻譯，請再次校對。"
        except DubbingError as exc:
            raise gr.Error(str(exc)) from exc

    def render(job, rows, reference, progress=PROGRESS):
        if job is None:
            raise gr.Error("請先上傳影片並完成辨識。")
        try:
            apply_edits(job, rows)
            before = {segment.id: segment.target for segment in job.segments}
            video, translated, source = render_job(job, reference, settings, progress)
            changed = [str(segment.id) for segment in job.segments if segment.target != before[segment.id]]
            status = "配音完成。可播放預覽，或下載影片與字幕。"
            if changed:
                status += f" 自動縮短第 {', '.join(changed)} 段；請校對更新後的配音譯文。原譯文已備份。"
            return job, table_rows(job), str(video), str(video), str(translated), str(source), status
        except SpeechTimingError as exc:
            return job, table_rows(job), None, None, None, None, str(exc)
        except DubbingError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="地端影片配音", analytics_enabled=False) as demo:
        job = gr.State(None)
        gr.Markdown("# 地端影片配音\n用中文錄 Demo，輸出英文配音；也支援英文轉中文。")
        gr.Markdown("影片在你的電腦處理。先上傳與校對，再產生聲音；原始影片另存保留。")
        with gr.Accordion("環境狀態與連線檢查", open=False):
            environment = gr.Textbox(label="檢查結果", lines=8, interactive=False)
            check = gr.Button("檢查環境")
            check.click(environment_status, outputs=environment, api_name=False)
        gr.Markdown("### 1. 上傳影片")
        with gr.Row():
            upload = gr.File(label="選擇影片（最長 5 分鐘，最大 1 GiB）", file_types=["video"], type="filepath")
            with gr.Column():
                direction = gr.Radio(list(DIRECTIONS), value="中文 → 英文", label="翻譯方向")
                prepare_button = gr.Button("辨識與翻譯", variant="primary")
        status = gr.Textbox(label="處理狀態", value="請先選擇影片。", interactive=False, lines=3)
        gr.Markdown("### 2. 校對逐句文字\n只修改「原文」與「配音譯文」，保留段落編號與順序。修改原文後，可重新翻譯。")
        rows = gr.Dataframe(
            headers=["編號", "原始時間", "原文", "配音譯文"], datatype=["str"] * 4,
            type="array", interactive=True,
            label="逐句校對", wrap=True, static_columns=[0, 1],
        )
        translate_button = gr.Button("依修正原文重新翻譯")
        gr.Markdown("### 3. 產生配音\n預設使用影片中的聲音作參考。也可提供你自己的 3–15 秒乾淨人聲，較長檔案只取前 15 秒。")
        reference = gr.File(label="參考聲音（選填）", file_types=["audio"], type="filepath")
        gr.Markdown("配音會替換整條原音軌，背景音樂及操作聲也會移除。超時段落會自動縮句並重配，最多兩輪；完成後請校對更新的譯文。")
        render_button = gr.Button("產生配音影片", variant="primary")
        preview = gr.Video(label="配音預覽", interactive=False)
        with gr.Row():
            download_video = gr.File(label="下載 MP4", interactive=False)
            download_translation = gr.File(label="下載譯文字幕 SRT", interactive=False)
            download_source = gr.File(label="下載原文字幕 SRT", interactive=False)

        prepare_button.click(
            prepare, inputs=[upload, direction],
            outputs=[job, rows, status, preview, download_video, download_translation, download_source],
            concurrency_id="models", concurrency_limit=1, api_name=False,
        )
        translate_button.click(
            retranslate, inputs=[job, rows], outputs=[job, rows, status],
            concurrency_id="models", concurrency_limit=1, api_name=False,
        )
        render_button.click(
            render, inputs=[job, rows, reference],
            outputs=[job, rows, preview, download_video, download_translation, download_source, status],
            concurrency_id="models", concurrency_limit=1, api_name=False,
        )
        # Invalidate stale work before allowing a different input to be processed.
        for component in (upload, direction):
            component.change(
                lambda: (None, [], None, None, None, None, "影片或翻譯方向已變更，請重新辨識。"),
                outputs=[job, rows, preview, download_video, download_translation, download_source, status],
                concurrency_id="models", concurrency_limit=1, api_name=False,
            )
        for event in (rows.input, reference.change):
            event(
                lambda: (None, None, None, None, "內容已變更，請重新產生配音。"),
                outputs=[preview, download_video, download_translation, download_source, status],
                concurrency_id="models", concurrency_limit=1, api_name=False,
            )
    return demo.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description="地端影片配音")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    args = parser.parse_args()
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("HTTPS 憑證與私鑰必須一起提供。")
    username, password = os.getenv("DUBBING_USERNAME"), os.getenv("DUBBING_PASSWORD")
    if bool(username) != bool(password):
        parser.error("DUBBING_USERNAME 與 DUBBING_PASSWORD 必須一起設定。")
    build_app().launch(
        server_name=args.host, server_port=args.port, share=False,
        auth=(username, password) if username else None,
        ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
        ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None,
        theme=gr.themes.Soft(primary_hue="indigo", font=["system-ui", "sans-serif"]), css=CSS,
        max_file_size="1gb", show_error=False,
    )


if __name__ == "__main__":
    main()
