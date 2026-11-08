import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../ui/chips';
import '../host/host.css';
import { App } from './App';

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
