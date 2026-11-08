// The address phones should open. A host page on "localhost" is only reachable from this PC, so
// the TV's join link and QR code use this machine's Wi-Fi address instead (same port).
import { useEffect, useState } from 'react';
import { serverInfo } from './api';

const LOCAL = ['localhost', '127.0.0.1', '[::1]'];

export function useJoinOrigin(): () => string {
  const [ip, setIp] = useState<string | null>(null);
  useEffect(() => {
    if (!LOCAL.includes(location.hostname)) return;
    serverInfo().then((i) => setIp(i.ips.find((a) => a.startsWith('192.168.') || a.startsWith('10.')) || i.ips[0] || null)).catch(() => {});
  }, []);
  return () => (ip ? `${location.protocol}//${ip}${location.port ? `:${location.port}` : ''}` : location.origin);
}
