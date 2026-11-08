import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../ui/chips';
import './host.css';
import { App } from './App';
import { reportErrors } from '../net/errors';

reportErrors('host');
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
