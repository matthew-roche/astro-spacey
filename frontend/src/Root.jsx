import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from 'react-router-dom';
import Shell from './Shell'
import './theme.css'

createRoot(document.getElementById("root")).render(<BrowserRouter><Shell/></BrowserRouter>);
