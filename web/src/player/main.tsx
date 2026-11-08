import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../ui/chips';
import './player.css';
import { App } from './App';
import { reportErrors } from '../net/errors';

reportErrors('player');
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
