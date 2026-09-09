import { useEffect, useState } from 'react';

type Status = { configured: boolean };

export default function ApiKeys() {
  const [key, setKey] = useState('');
  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  async function request(method = 'GET', value?: string) {
    const response = await fetch('/api/credentials/openrouter', {
      method, headers: { 'Content-Type': 'application/json' },
      body: value === undefined ? undefined : JSON.stringify({ key: value }),
    });
    if (!response.ok) throw new Error('Не удалось выполнить операцию с ключом. Проверьте backend и введённое значение.');
    return response.json() as Promise<Status>;
  }
  useEffect(() => { let live = true; request().then(s => { if (live) setStatus(s); }).catch(() => { if (live) setError('Статус ключа недоступен: проверьте backend.'); }); return () => { live = false; }; }, []);
  async function change(remove: boolean) {
    setBusy(true); setError(''); setMessage('');
    const value = key.trim();
    setKey('');
    try {
      setStatus(await request(remove ? 'DELETE' : 'PUT', remove ? undefined : value));
      setMessage(remove ? 'Ключ удалён из серверного хранилища.' : 'Ключ сохранён. Проверка у провайдера не выполнялась.');
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  }
  return <section aria-label="AI API keys">
    <h3>OpenRouter API key <span className="tag">{status === null ? 'Статус неизвестен' : status.configured ? 'Сохранён · не проверен' : 'Не задан'}</span></h3>
    <p>Ключ шифруется Windows DPAPI на сервере для текущего пользователя Windows. В экспорт и localStorage он не включается.</p>
    <form onSubmit={e => { e.preventDefault(); void change(false); }}>
      <label className="field"><span>{status?.configured ? 'Новый ключ для замены' : 'API key'}</span>
        <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={e => setKey(e.target.value)} placeholder="Вставьте ключ OpenRouter" disabled={busy} minLength={16} maxLength={4096} required />
      </label>
      <button type="submit" disabled={busy || key.trim().length < 16}>{busy ? 'Сохранение…' : status?.configured ? 'Заменить ключ' : 'Сохранить ключ'}</button>
      <button type="button" disabled={busy || !status?.configured} onClick={() => void change(true)}>Удалить ключ</button>
    </form>
    {error && <p role="alert">{error}</p>}{message && <p role="status">{message}</p>}
    <p>Сохранение не делает запросов к OpenRouter. Платные прогоны отключены; исполнитель ИИ пока не подключён.</p>
  </section>;
}
