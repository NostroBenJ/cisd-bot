import json, time
from datetime import date, timedelta
from research.uw_recorder import api_key, fetch, _rows, STORE
k = api_key()
have = {p.stem for p in (STORE/'SPY_ohlc_1m').glob('*.json')}
d, end = date(2022,1,3), date(2025,8,15)
todo = []
while d <= end:
    if d.weekday() < 5 and d.isoformat() not in have:
        todo.append(d.isoformat())
    d += timedelta(days=1)
print(f'have {len(have)} days; fetching {len(todo)}', flush=True)
got = empty = 0
for i, day in enumerate(todo, 1):
    st, pl = fetch('/api/stock/SPY/ohlc/1m', k, date=day, limit=2500)
    n = _rows(pl) if st == 'ok' else 0
    if n:
        (STORE/'SPY_ohlc_1m'/f'{day}.json').write_text(json.dumps(pl), encoding='utf-8'); got += 1
    else:
        empty += 1
    if i % 100 == 0:
        print(f'  [{i}/{len(todo)}] {day}  got {got}  empty {empty}', flush=True)
    time.sleep(0.08)
print(f'done: {got} fetched, {empty} empty. total {len(list((STORE/"SPY_ohlc_1m").glob("*.json")))} days', flush=True)
