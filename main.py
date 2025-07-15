from sentence_transformers import SentenceTransformer
import numpy as np

import re
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set
from pymorphy3 import MorphAnalyzer
from collections import defaultdict

from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SmartSummaryGenerator:
    def __init__(self):
        self.morph = MorphAnalyzer()

    def prepare_text(self, text: str) -> str:
        if not text:
            return ""
        # Удаляем ссылки, упоминания и лишние пробелы, но сохраняем структуру и эмодзи
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def find_sentences(self, text: str) -> List[str]:
        sentences = []
        current = []
        for token in re.findall(r'\w+|[.,!?]', text):
            if token in {'.', '!', '?'}:
                if current:
                    sentences.append(' '.join(current) + token)
                    current = []
            else:
                current.append(token)
        if current:
            sentences.append(' '.join(current))
        return [s for s in sentences if len(s.split()) >= 2]

    def _normalize_phrase(self, phrase: str) -> str:
        words = [self.morph.parse(w.lower())[0].normal_form for w in phrase.split()]
        return ' '.join(words)

    
    def generate_summary(self, text: str, keywords: List[str]) -> str:
        text = self.prepare_text(text)
        if not text:
            return "Нет текста для анализа"

        sentences = self.find_sentences(text)
        if not sentences:
            return text[:500] + ("..." if len(text) > 500 else "")

        summary = [sentences[0]]  # Всегда включаем первое предложение
        used_indices = {0}
        found_keywords = set()

        normalized_phrases = {self._normalize_phrase(kw): kw for kw in keywords if ' ' in kw}
        normalized_words = {
            self.morph.parse(kw.lower())[0].normal_form: kw
            for kw in keywords
            if ' ' not in kw
        }

        for i, sent in enumerate(sentences[1:], 1):
            sent_normalized = self._normalize_phrase(sent)
            matched = False
            for norm_phrase, orig_phrase in normalized_phrases.items():
                if norm_phrase in sent_normalized:
                    summary.append(sent)
                    used_indices.add(i)
                    found_keywords.add(orig_phrase)
                    matched = True
                    break
            if matched:
                continue

            for word in re.findall(r'\w+', sent.lower()):
                norm_word = self.morph.parse(word)[0].normal_form
                if norm_word in normalized_words:
                    summary.append(sent)
                    used_indices.add(i)
                    found_keywords.add(normalized_words[norm_word])
                    break

            if len(summary) >= 5:
                break

        if len(summary) < 5:
            for i, sent in enumerate(sentences):
                if i in used_indices:
                    continue
                summary.append(sent)
                if len(summary) >= 5:
                    break

        return ' '.join(summary[:5])
class TelegramMonitor:
    def __init__(self):
        self.api_id = 
        self.api_hash = 
        self.session_string = 
        self.bot_token = 
        self.target_channel = 
        self.channels = "rbc_news, bankoffo, kommersant, ejdailyru, banksta, ROIrosinkas, tbank, interfaxonline, plusjournal, bankiruofficial, profindustrycom, gazprombank, cccbnews, AlfaBank, bankvtb, vedomosti, rshb_tg, minfin, rosstat_official, minec_russia"
        self.keywords = "инкассация, купюра, спецавтомобиль, кассовый центр, обслуживание платежных терминалов, перевозка ценностей, депозитные машины, перевозка и доставка денежных средств, транспортировка денежных средств, монеты, банкноты, денежное обращение, инкассаторские перевозки, инкассаторский центр, доставка наличных, инкассация терминалов, маршруты инкассации, инкассаторская компания, хранение ценностей, хранение слитков, перевозка драгметаллов, перевозка кассет, перевозка сейф-пакетов, загрузка банкоматов, выгрузка банкоматов, доставка денег в банк, рынок инкассации, платежный бизнес, инкассатор, касса пересчета, онлайн-инкассация, самоинкассация, перевозка наличных денег, пересчет наличных, подкрепление офисов, подкрепление отделений, доставка размена, рассчетно-кассовые центры банка россии, кассовое обслуживание, ркц, сокращение наличных, наличные, автоматизированные депозитные машины, наличная валюта, наличные рубли"
        self.max_manual_days = 7
        self.auto_search_days = 1
        self.auto_search_time = {'hour': 7, 'minute': 0}
        self.message_delay = 0.3

        self.client = TelegramClient(
            StringSession(self.session_string),
            self.api_id,
            self.api_hash
        )
        self.bot = Bot(token=self.bot_token)
        self.dp = Dispatcher()
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.summarizer = SmartSummaryGenerator()
        self.active_tasks = set()
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.seen_post_embeddings = []
        self.similarity_threshold = 0.85
        self.seen_posts: Set[str] = set()

        self.dp.message.register(self.handle_parse, Command("parse"))

    async def safe_send_message(self, chat_id: str, text: str):
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Ошибка отправки в {chat_id}: {e}")

    def _get_post_fingerprint(self, text: str) -> str:
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        return cleaned[:100]

    def _contains_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        found = set()
        text_lower = text.lower()
        text_normalized = ' '.join([
            self.summarizer.morph.parse(w)[0].normal_form
            for w in re.findall(r'\w+', text_lower)
        ])
        single_words = []
        multiword_phrases = []
        for kw in self.keywords.split(','):
            kw = kw.strip()
            if not kw:
                continue
            if ' ' in kw:
                multiword_phrases.append(kw)
            else:
                single_words.append(kw)

        for phrase in multiword_phrases:
            if phrase.lower() in text_lower:
                found.add(phrase)
                continue
            norm_phrase = self.summarizer._normalize_phrase(phrase)
            if norm_phrase in text_normalized:
                found.add(phrase)

        for word in single_words:
            if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
                found.add(word)
                continue
            norm_word = self.summarizer.morph.parse(word.lower())[0].normal_form
            if re.search(r'\b' + re.escape(norm_word) + r'\b', text_normalized):
                found.add(word)
        return list(found)

    async def _scan_channel(self, channel: str, days: int) -> List[Dict]:
        results = []
        date_limit = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            entity = await self.client.get_entity(channel)
            async for msg in self.client.iter_messages(entity, offset_date=date_limit, reverse=True):
                try:
                    text = getattr(msg, 'text', '') or ''
                    if not text.strip():
                        continue
                    post_fingerprint = self._get_post_fingerprint(text)
                    if post_fingerprint in self.seen_posts:
                        continue
                    keywords = self._contains_keywords(text)
                    if not keywords:
                        continue
                    embedding = self.embedding_model.encode(text)
                    is_duplicate = False
                    for emb in self.seen_post_embeddings:
                        sim = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
                        if sim >= self.similarity_threshold:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue
                    self.seen_post_embeddings.append(embedding)
                    summary = self.summarizer.generate_summary(text, keywords)
                    results.append({
                        "channel": channel,
                        "date": msg.date.strftime('%d.%m.%Y %H:%M'),
                        "keywords": ", ".join(keywords),
                        "summary": summary,
                        "link": f"https://t.me/{channel}/{msg.id}",
                        "fingerprint": post_fingerprint
                    })
                    self.seen_posts.add(post_fingerprint)
                except Exception as e:
                    logger.error(f"Ошибка обработки сообщения: {e}")
                    continue
        except Exception as e:
            logger.error(f"Ошибка сканирования {channel}: {e}")
        return results
    async def search_all_channels(self, days: int) -> List[Dict]:
        if not self.client.is_connected():
            await self.client.connect()
        self.seen_posts.clear()
        self.seen_post_embeddings.clear()
        channels = [ch.strip() for ch in self.channels.split(',') if ch.strip()]
        tasks = []
        for channel in channels:
            task = asyncio.create_task(self._scan_channel(channel, days))
            tasks.append(task)
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
        all_results = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.error(f"Ошибка парсинга: {result}")
            elif result:
                all_results.extend(result)
        return sorted(all_results, key=lambda x: x["date"], reverse=True)

    def _format_result(self, data: Dict, idx: int = None, total: int = None) -> str:
        header = f"🔍 Результат {idx}/{total}\n" if idx and total else ""
        return (
            f"{header}"
            f"📌 Канал: @{data['channel']}\n"
            f"📅 {data['date']}\n"
            f"🔑 Ключевые слова: {data['keywords']}\n"
            f"📝 {data['summary']}\n"
            f"🔗 {data['link']}"
        )

    async def handle_parse(self, message: types.Message):
        try:
            args = message.text.split()
            days = int(args[1]) if len(args) > 1 and args[1].isdigit() else self.auto_search_days
            if days > self.max_manual_days:
                await message.answer(f"❌ Максимальный период - {self.max_manual_days} дней")
                return
            await message.answer(
                f"🔍 Ищу посты за последние {days} дней\n"
                f"📌 Каналы: {self.channels}\n"
                f"🔑 Ключевые слова: {self.keywords}"
            )
            results = await self.search_all_channels(days)
            if not results:
                await message.answer("ℹ️ Сообщений не найдено")
                return
            total = len(results)
            for idx, item in enumerate(results, 1):
                await self.safe_send_message(
                    message.chat.id,
                    self._format_result(item, idx, total)
                )
                await asyncio.sleep(self.message_delay)
            await message.answer(f"✅ Найдено {total} уникальных сообщений")
        except Exception as e:
            logger.error(f"Ошибка /parse: {e}")
            await message.answer("⚠️ Ошибка поиска")

    async def auto_search(self):
        try:
            logger.info("Запуск автоматического поиска...")
            print("⌛ Запуск автоматического поиска...")
            results = await self.search_all_channels(self.auto_search_days)
            if not results:
                await self.safe_send_message(self.target_channel, "ℹ️ Нет новых сообщений с ключевыми словами за сутки")
                print("ℹ️ Нет новых сообщений с ключевыми словами за сутки")
                return
            stats = defaultdict(int)
            for post in results:
                stats[post["channel"]] += 1
            report = (
                "📊 Ежедневный мониторинг\n"
                f"📅 Период: прошедшие сутки\n"
                f"🔍 Найдено: {len(results)} пост\n\n"
                "📈 Статистика:\n" +
                "\n".join(f"• @{ch}: {cnt}" for ch, cnt in stats.items())
            )
            await self.safe_send_message(self.target_channel, report)
            for post in results:
                await self.safe_send_message(
                    self.target_channel,
                    self._format_result(post)
                )
                await asyncio.sleep(self.message_delay)
            print(f"✅ Автопоиск завершен. Найдено {len(results)} уникальных сообщений")
        except Exception as e:
            logger.error(f"Ошибка авто-поиска: {e}")
            print(f"⚠️ Ошибка авто-поиска: {e}")

    async def run(self):
        try:
            print("🟢 Мониторинг активирован")
            print(f"📌 Каналы: {self.channels}")
            print(f"🔑 Ключевые слова: {self.keywords}")
            print(f"⏰ Автопоиск в {self.auto_search_time['hour']}:{self.auto_search_time['minute']:02d} UTC")
            await self.client.start()
            if not await self.client.is_user_authorized():
                print("❌ Ошибка авторизации в Telegram")
                return
            self.scheduler.add_job(
                self.auto_search,
                'cron',
                **self.auto_search_time
            )
            self.scheduler.start()
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Ошибка запуска: {e}")
            print(f"❌ Ошибка запуска: {e}")
        finally:
            print("🔴 Мониторинг остановлен")
            if self.scheduler.running:
                self.scheduler.shutdown()
            if self.client.is_connected():
                await self.client.disconnect()

monitor = TelegramMonitor()

async def start_bot():
    await monitor.run()

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.create_task(start_bot())
        print("Бот запущен. Используйте /parse [дни] (макс. 7 дней)")
    else:
        asyncio.run(start_bot())
except Exception as e:
    logger.error(f"Ошибка запуска: {e}")
    print(f"❌ Ошибка запуска: {e}")