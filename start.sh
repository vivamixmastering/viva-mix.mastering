#!/usr/bin/env bash
# start.sh — رندر و ریلوی این فایل رو اجرا می‌کنن
# یک «زنده نگهدارنده» ساده وب + ربات تلگرام در پس‌زمینه

echo "▶️ شروع MixMaster Bot..."

PORT="${PORT:-8000}"

# اگر خاموشه، فقط وب زنده می‌مونه و ربات اجرا نمی‌شه (حالت خواب)
if [ "${BOT_POWER:-on}" = "off" ]; then
  echo "😴 BOT_POWER=off — ربات در حالت خوابه. فقط سرور وب روشن می‌مونه."
  python -m http.server "$PORT" --directory . &
  exec sleep infinity
fi

# ربات تلگرام در پس‌زمینه
python bot.py &
BOT_PID=$!
echo "🤖 ربات با PID=$BOT_PID اجرا شد"

# زنده نگهدارنده وب (برای پلتفرم ابری)
python -m http.server "$PORT" --directory . &

# اگه ربات بخوابه، کل پروسه رو بنداز پایین تا پلتفرم ری‌استارتش کنه
wait $BOT_PID
exit 1
