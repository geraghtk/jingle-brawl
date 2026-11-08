import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../ui/chips';
import '../host/host.css';
import { App } from './App';
import { reportErrors } from '../net/errors';

reportErrors('tv');
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
