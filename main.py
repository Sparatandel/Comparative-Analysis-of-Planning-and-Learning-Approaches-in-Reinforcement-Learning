"""
╔══════════════════════════════════════════════════════════════════╗
║  Comparative Analysis of Planning and Learning Approaches        ║
║  in Reinforcement Learning Using Grid World Environments         ║
║  ── Dark Knight Edition ──                                       ║
╚══════════════════════════════════════════════════════════════════╝

Main Application Entry Point
Run: python main.py
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import io
import os

port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)

# ── Color Palette ─────────────────────────────────────────────────
BG_DEEP   = '#08080e'
BG_BASE   = '#0d0d18'
BG_PANEL  = '#12121f'
BG_CARD   = '#191928'
BG_HOVER  = '#1e1e32'
BORDER    = '#252538'
BORDER_HI = '#3a3a5c'

GOLD      = '#c9a84c'
GOLD_LT   = '#e8c86a'
GOLD_DIM  = '#7a6030'
BLUE      = '#2a6dd9'
BLUE_LT   = '#4a8df0'
CRIMSON   = '#d9322a'
CRIMSON_LT= '#f05050'

TEXT_HI   = '#f0f0fa'
TEXT_MID  = '#b0b0c8'
TEXT_DIM  = '#606080'

SUCCESS   = '#3aad6a'
WARNING   = '#d9aa2a'

ALGO_COLORS = {
    'Value Iteration': GOLD,
    'Q-Learning':      BLUE,
    'SARSA':           CRIMSON,
}

ALGO_DESC = {
    'Value Iteration': 'Model-Based Planning\nUses full MDP knowledge\nDynamic Programming approach',
    'Q-Learning':      'Off-Policy TD Learning\nModel-free, learns from experience\nMaximizes future Q-values',
    'SARSA':           'On-Policy TD Learning\nModel-free, follows actual policy\nConservative, safer paths',
}


def pil_to_tk(pil_img):
    return ImageTk.PhotoImage(pil_img)


def buf_to_pil(buf, size=None):
    buf.seek(0)
    img = Image.open(buf)
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


class DKButton(tk.Canvas):
    """Custom Dark Knight styled button with hover glow effect"""
    
    def __init__(self, parent, text, command=None, color=GOLD, width=200, height=44,
                 font_size=11, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=BG_PANEL, highlightthickness=0, cursor='hand2', **kwargs)
        self.command = command
        self.color = color
        self._btn_width = width
        self._btn_height = height
        self.text = text
        self.font_size = font_size
        self._hover = False
        self._press = False
        self._anim_alpha = 0
        
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<ButtonPress-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)
        
        self.after_idle(self._draw)
    
    def _hex_to_rgb(self, h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    def _lerp_color(self, c1, c2, t):
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _draw(self):
        self.delete('all')
        w, h = self._btn_width, self._btn_height
        t = self._anim_alpha
        
        bg_color = self._lerp_color(BG_CARD, BG_HOVER, t)
        border_color = self._lerp_color(BORDER_HI, self.color, t)
        text_color = self._lerp_color(TEXT_MID, TEXT_HI, t)
        
        # Shadow glow on hover
        if t > 0.1:
            for i in range(4, 0, -1):
                alpha_i = t * (i / 4) * 0.5
                gc = self._lerp_color(BG_BASE, self.color, alpha_i * 0.4)
                self.create_rectangle(i, i, w-i, h-i, 
                                     outline=gc, fill='', width=1)
        
        # Main body
        self.create_rectangle(2, 2, w-2, h-2, 
                             outline=border_color, fill=bg_color, width=1)
        
        # Accent line at top
        accent = self._lerp_color(BG_CARD, self.color, t * 0.7)
        self.create_line(4, 2, w-4, 2, fill=accent, width=1)
        
        # Text
        if self._press:
            text_color = self.color
        self.create_text(w//2, h//2, text=self.text, fill=text_color,
                        font=('Courier New', self.font_size, 'bold'))
    
    def _animate(self, target, step=0.12):
        if abs(self._anim_alpha - target) < 0.05:
            self._anim_alpha = target
            self._draw()
            return
        self._anim_alpha += step if target > self._anim_alpha else -step
        self._draw()
        self.after(16, lambda: self._animate(target, step))
    
    def _on_enter(self, e):
        self._hover = True
        self._animate(1.0)
    
    def _on_leave(self, e):
        self._hover = False
        self._press = False
        self._animate(0.0)
    
    def _on_press(self, e):
        self._press = True
        self._draw()
    
    def _on_release(self, e):
        self._press = False
        self._draw()
        if self.command and self._hover:
            self.command()


class DKLabel(tk.Label):
    def __init__(self, parent, text, color=TEXT_MID, size=10, bold=False, **kwargs):
        weight = 'bold' if bold else 'normal'
        super().__init__(parent, text=text, fg=color, bg=kwargs.pop('bg', BG_PANEL),
                        font=('Courier New', size, weight), **kwargs)


class DKFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=kwargs.pop('bg', BG_PANEL),
                        highlightthickness=1,
                        highlightbackground=BORDER, **kwargs)


class AlgoCard(tk.Frame):
    """Selectable algorithm card with hover animation"""
    
    def __init__(self, parent, name, desc, color, on_select, **kwargs):
        super().__init__(parent, bg=BG_CARD, bd=0,
                        highlightthickness=2, highlightbackground=BORDER,
                        cursor='hand2', **kwargs)
        self.name = name
        self.color = color
        self.on_select = on_select
        self._selected = False
        
        # Color bar
        bar = tk.Frame(self, bg=color, height=3)
        bar.pack(fill='x', side='top')
        
        inner = tk.Frame(self, bg=BG_CARD, padx=16, pady=12)
        inner.pack(fill='both', expand=True)
        
        tk.Label(inner, text=name, fg=color, bg=BG_CARD,
                font=('Courier New', 13, 'bold')).pack(anchor='w')
        
        tk.Label(inner, text=desc, fg=TEXT_DIM, bg=BG_CARD,
                font=('Courier New', 9), justify='left').pack(anchor='w', pady=(4, 0))
        
        self._bind_all(self)
    
    def _bind_all(self, widget):
        widget.bind('<Enter>', self._on_enter)
        widget.bind('<Leave>', self._on_leave)
        widget.bind('<Button-1>', self._on_click)
        for child in widget.winfo_children():
            self._bind_all(child)
    
    def _on_enter(self, e):
        if not self._selected:
            self.configure(highlightbackground=self.color)
    
    def _on_leave(self, e):
        if not self._selected:
            self.configure(highlightbackground=BORDER)
    
    def _on_click(self, e):
        self.on_select(self.name)
    
    def set_selected(self, val):
        self._selected = val
        if val:
            self.configure(highlightbackground=self.color, bg=BG_HOVER)
        else:
            self.configure(highlightbackground=BORDER, bg=BG_CARD)


class GridGameCanvas(tk.Canvas):
    """Interactive GridWorld game canvas"""
    
    CELL = 58
    
    def __init__(self, parent, env, agent, algo_name, on_step_callback=None, **kwargs):
        self.env = env
        self.agent = agent
        self.algo_name = algo_name
        self.on_step_callback = on_step_callback
        self.color = ALGO_COLORS.get(algo_name, GOLD)
        
        sz = env.size
        cw = sz * self.CELL + 2
        self._canvas_size = cw
        super().__init__(parent, width=cw, height=cw,
                        bg=BG_BASE, highlightthickness=1,
                        highlightbackground=BORDER, **kwargs)
        
        self.player_pos = env.start
        self.path = [env.start]
        self.game_over = False
        self.won = False
        self.steps = 0
        self.total_reward = 0.0
        
        self.after_idle(self._draw_all)
        self.after_idle(self.focus_set)
        self.bind('<KeyPress>', self._on_key)
    
    def _cell_xy(self, r, c):
        x = c * self.CELL + 1
        y = r * self.CELL + 1
        return x, y
    
    def _draw_all(self):
        self.delete('all')
        env = self.env
        sz = env.size
        C = self.CELL
        
        for r in range(sz):
            for c in range(sz):
                x, y = self._cell_xy(r, c)
                pos = (r, c)
                
                if pos in env.obstacles:
                    self.create_rectangle(x, y, x+C, y+C, fill='#0a0a12', outline='#0d0d1a', width=1)
                    # Texture lines
                    self.create_line(x+4, y+4, x+C-4, y+C-4, fill='#111120', width=1)
                    self.create_line(x+C-4, y+4, x+4, y+C-4, fill='#111120', width=1)
                elif pos == env.goal:
                    self.create_rectangle(x, y, x+C, y+C, fill='#1a1a0a', outline=GOLD, width=1)
                    self.create_text(x+C//2, y+C//2, text='★', fill=GOLD, font=('Arial', 20))
                elif pos == env.start:
                    self.create_rectangle(x, y, x+C, y+C, fill='#0a0a1e', outline=BLUE, width=1)
                else:
                    fill = '#0f0f1e' if (r+c)%2==0 else '#0d0d1a'
                    self.create_rectangle(x, y, x+C, y+C, fill=fill, outline='#0e0e1c', width=1)
        
        # Draw path
        for i, pos in enumerate(self.path[:-1]):
            r, c = pos
            x, y = self._cell_xy(r, c)
            alpha = 0.3 + 0.5 * (i / max(len(self.path), 1))
            self.create_rectangle(x+2, y+2, x+C-2, y+C-2,
                                 fill='#1a1a35', outline='', width=0)
        
        # Player
        if not self.game_over:
            r, c = self.player_pos
            x, y = self._cell_xy(r, c)
            cx, cy = x + C//2, y + C//2
            # Glow
            self.create_oval(cx-18, cy-18, cx+18, cy+18, fill='#1a1a40', outline='', width=0)
            self.create_oval(cx-13, cy-13, cx+13, cy+13, fill=self.color, outline='', width=0)
            self.create_text(cx, cy, text='◆', fill=BG_BASE, font=('Arial', 14, 'bold'))
        
        # AI suggestion arrow
        if not self.game_over and hasattr(self.agent, 'get_action'):
            ai_action = self.agent.get_action(self.player_pos)
            arrow_chars = {0:'↑', 1:'↓', 2:'←', 3:'→'}
            r, c = self.player_pos
            x, y = self._cell_xy(r, c)
            self.create_text(x + C - 8, y + 8, text=arrow_chars[ai_action],
                           fill=self.color, font=('Arial', 10, 'bold'))
        
        if self.won:
            cs = self._canvas_size
            self.create_rectangle(0, 0, cs, cs,
                                 fill='', outline=GOLD, width=4)
            self.create_text(cs//2, cs//2,
                           text='GOAL\nREACHED', fill=GOLD, font=('Courier New', 20, 'bold'),
                           justify='center')
    
    def _on_key(self, e):
        if self.game_over:
            return
        
        key_action = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3,
                     'w': 0, 's': 1, 'a': 2, 'd': 3}
        
        action = key_action.get(e.keysym)
        if action is None:
            return
        
        old_pos = self.player_pos
        dr, dc = self.env.ACTIONS[action]
        nr, nc = old_pos[0] + dr, old_pos[1] + dc
        
        if self.env.is_valid((nr, nc)):
            self.player_pos = (nr, nc)
            self.env.state = (nr, nc)
        
        reward = -1.0
        done = False
        if self.player_pos == self.env.goal:
            reward = 100.0
            done = True
            self.won = True
        
        self.steps += 1
        self.total_reward += reward
        self.path.append(self.player_pos)
        
        if done:
            self.game_over = True
        
        if self.on_step_callback:
            self.on_step_callback(self.steps, self.total_reward, self.player_pos)
        
        self._draw_all()
    
    def auto_play(self):
        """Let AI play automatically"""
        if self.game_over:
            return
        action = self.agent.get_action(self.player_pos)
        dr, dc = self.env.ACTIONS[action]
        nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
        
        if self.env.is_valid((nr, nc)):
            self.player_pos = (nr, nc)
        
        reward = -1.0
        done = False
        if self.player_pos == self.env.goal:
            reward = 100.0
            done = True
            self.won = True
        
        self.steps += 1
        self.total_reward += reward
        self.path.append(self.player_pos)
        
        if done:
            self.game_over = True
        
        if self.on_step_callback:
            self.on_step_callback(self.steps, self.total_reward, self.player_pos)
        
        self._draw_all()
        
        if not self.game_over and self.steps < self.env.max_steps:
            self.after(120, self.auto_play)
    
    def reset_game(self):
        self.player_pos = self.env.start
        self.env.reset()
        self.path = [self.env.start]
        self.game_over = False
        self.won = False
        self.steps = 0
        self.total_reward = 0.0
        self._draw_all()


class ProgressBar(tk.Canvas):
    def __init__(self, parent, width=400, height=14, color=GOLD, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=BG_CARD, highlightthickness=0, **kwargs)
        self._bar_w = width
        self._bar_h = height
        self._color = color
        self._pct = 0
        self.after_idle(self._draw)
    
    def set(self, pct):
        self._pct = max(0, min(1, pct))
        self._draw()
    
    def _draw(self):
        self.delete('all')
        self.create_rectangle(0, 0, self._bar_w, self._bar_h, fill='#0d0d1a', outline='#1a1a2a')
        if self._pct > 0:
            fw = int(self._pct * (self._bar_w - 2))
            self.create_rectangle(1, 1, 1+fw, self._bar_h-1, fill=self._color, outline='')
            # Shine
            self.create_rectangle(1, 1, 1+fw, 4, fill='#ffffff20', outline='')


#   MAIN APPLICATION

class DarkKnightRL(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title('RL Comparative Analysis')
        self.configure(bg=BG_DEEP)
        self.geometry('1320x820')
        self.minsize(1100, 700)
        
        # State
        self.selected_algo = None
        self.env = None
        self.agents = {}
        self.metrics = {}
        self.training_done = False
        self.current_page = 'home'
        
        self._build_ui()
        self._show_page('home')
    
    # ── UI Building ───────────────────────────────────────────────
    
    def _build_ui(self):
        # Top navigation bar
        self._build_navbar()
        
        # Page container
        self.page_frame = tk.Frame(self, bg=BG_DEEP)
        self.page_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Build pages
        self.pages = {}
        self._build_home_page()
        self._build_select_page()
        self._build_game_page()
        self._build_analysis_page()
        self._build_about_page()
    
    def _build_navbar(self):
        nav = tk.Frame(self, bg=BG_PANEL, height=54)
        nav.pack(fill='x', side='top')
        nav.pack_propagate(False)
        
        # Left: logo
        logo_frame = tk.Frame(nav, bg=BG_PANEL)
        logo_frame.pack(side='left', padx=20, pady=8)
        
        tk.Label(logo_frame, text='◈', fg=GOLD, bg=BG_PANEL,
                font=('Arial', 18)).pack(side='left', padx=(0, 8))
        tk.Label(logo_frame, text='RL ANALYSIS', fg=TEXT_HI, bg=BG_PANEL,
                font=('Courier New', 13, 'bold')).pack(side='left')
        tk.Label(logo_frame, text=' // DARK KNIGHT EDITION', fg=GOLD_DIM, bg=BG_PANEL,
                font=('Courier New', 9)).pack(side='left', pady=4)
        
        # Right: nav buttons
        nav_right = tk.Frame(nav, bg=BG_PANEL)
        nav_right.pack(side='right', padx=16, pady=8)
        
        self.nav_buttons = {}
        nav_items = [('home','HOME'), ('select','PLAY'), ('analysis','ANALYSIS'), ('about','ABOUT')]
        
        for page, label in nav_items:
            btn = tk.Label(nav_right, text=label, fg=TEXT_DIM, bg=BG_PANEL,
                          font=('Courier New', 10, 'bold'), cursor='hand2', padx=16, pady=4)
            btn.pack(side='left')
            btn.bind('<Enter>', lambda e, b=btn: b.configure(fg=GOLD))
            btn.bind('<Leave>', lambda e, b=btn, pg=page: 
                     b.configure(fg=GOLD if self.current_page == pg else TEXT_DIM))
            btn.bind('<Button-1>', lambda e, pg=page: self._show_page(pg))
            self.nav_buttons[page] = btn
        
        # Separator
        tk.Frame(nav, bg=GOLD_DIM, height=2).pack(fill='x', side='bottom')
    
    def _build_home_page(self):
        page = tk.Frame(self.page_frame, bg=BG_DEEP)
        self.pages['home'] = page
        
        # Hero section
        hero = tk.Frame(page, bg=BG_DEEP)
        hero.pack(fill='both', expand=True, pady=(60, 20))
        
        # Bat symbol decoration (ASCII art)
        bat_art = """
         ██████╗  ██╗          ██████╗ ██████╗ ███╗   ███╗██████╗  █████╗ ██████╗ 
        ██╔══██╗ ██║         ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔══██╗
        ██████╔╝ ██║         ██║     ██║   ██║██╔████╔██║██████╔╝███████║██████╔╝
        ██╔══██╗ ██║         ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██║██╔══██╗
        ██║  ██║ ███████╗    ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║██║  ██║
        ╚═╝  ╚═╝ ╚══════╝     ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝"""
        
        tk.Label(hero, text=bat_art, fg='#2a2a3a', bg=BG_DEEP,
                font=('Courier New', 7)).pack(pady=(0, 0))
        
        tk.Label(hero, text='REINFORCEMENT LEARNING', fg=GOLD, bg=BG_DEEP,
                font=('Courier New', 28, 'bold')).pack()
        
        tk.Label(hero, text='Comparative Analysis of Planning and Learning Approaches',
                fg=TEXT_HI, bg=BG_DEEP, font=('Courier New', 14)).pack(pady=(4, 2))
        
        tk.Label(hero, text='Using Grid World Environments',
                fg=TEXT_MID, bg=BG_DEEP, font=('Courier New', 12)).pack()
        
        # Separator line with ornament
        sep_frame = tk.Frame(hero, bg=BG_DEEP)
        sep_frame.pack(pady=28)
        tk.Frame(sep_frame, bg=GOLD_DIM, width=120, height=1).pack(side='left')
        tk.Label(sep_frame, text='  ◈  ', fg=GOLD, bg=BG_DEEP, 
                font=('Arial', 14)).pack(side='left')
        tk.Frame(sep_frame, bg=GOLD_DIM, width=120, height=1).pack(side='left')
        
        # Three algorithm cards
        cards_frame = tk.Frame(hero, bg=BG_DEEP)
        cards_frame.pack(pady=10)
        
        algo_info = [
            ('Value Iteration', GOLD, 'Model-Based\nPlanning', 'Dynamic Programming\nFull MDP knowledge'),
            ('Q-Learning', BLUE, 'Off-Policy\nTD Learning', 'Model-Free\nMax Q-value updates'),
            ('SARSA', CRIMSON, 'On-Policy\nTD Learning', 'Model-Free\nActual policy updates'),
        ]
        
        for name, color, tag, sub in algo_info:
            card = tk.Frame(cards_frame, bg=BG_CARD, padx=20, pady=16,
                          highlightthickness=1, highlightbackground=BORDER, width=240)
            card.pack(side='left', padx=10)
            card.pack_propagate(False)
            
            tk.Frame(card, bg=color, height=2).pack(fill='x')
            tk.Label(card, text=name, fg=color, bg=BG_CARD,
                    font=('Courier New', 12, 'bold')).pack(pady=(8,2))
            tk.Label(card, text=tag, fg=TEXT_HI, bg=BG_CARD,
                    font=('Courier New', 10), justify='center').pack()
            tk.Label(card, text=sub, fg=TEXT_DIM, bg=BG_CARD,
                    font=('Courier New', 8), justify='center').pack(pady=(4,0))
        
        # CTA Buttons
        btn_frame = tk.Frame(hero, bg=BG_DEEP)
        btn_frame.pack(pady=30)
        
        DKButton(btn_frame, '▶  START PLAYING', 
                command=lambda: self._show_page('select'),
                color=GOLD, width=220, height=48, font_size=12).pack(side='left', padx=10)
        
        DKButton(btn_frame, '◉  VIEW ANALYSIS',
                command=self._run_analysis_and_show,
                color=BLUE, width=220, height=48, font_size=12).pack(side='left', padx=10)
        
        # Bottom stat bar
        stats_bar = tk.Frame(page, bg=BG_PANEL, height=40)
        stats_bar.pack(fill='x', side='bottom')
        
        stats = [('ALGORITHMS', '3'), ('GRID SIZE', '8×8'), 
                 ('METHOD', 'DP + TD'), ('FRAMEWORK', 'Python')]
        for label, val in stats:
            tk.Label(stats_bar, text=f'{label}: ', fg=TEXT_DIM, bg=BG_PANEL,
                    font=('Courier New', 9)).pack(side='left', padx=(20, 0), pady=10)
            tk.Label(stats_bar, text=val, fg=GOLD, bg=BG_PANEL,
                    font=('Courier New', 9, 'bold')).pack(side='left', pady=10)
            tk.Label(stats_bar, text='  │  ', fg=BORDER_HI, bg=BG_PANEL,
                    font=('Courier New', 9)).pack(side='left', pady=10)
    
    def _build_select_page(self):
        page = tk.Frame(self.page_frame, bg=BG_DEEP)
        self.pages['select'] = page
        
        # Header
        hdr = tk.Frame(page, bg=BG_DEEP)
        hdr.pack(fill='x', padx=40, pady=(30, 20))
        tk.Label(hdr, text='SELECT ALGORITHM & PLAY', fg=GOLD, bg=BG_DEEP,
                font=('Courier New', 20, 'bold')).pack(anchor='w')
        tk.Label(hdr, text='Choose an algorithm to play the GridWorld game. Compare strategies in real-time.',
                fg=TEXT_DIM, bg=BG_DEEP, font=('Courier New', 10)).pack(anchor='w', pady=(4, 0))
        tk.Frame(hdr, bg=GOLD_DIM, height=1).pack(fill='x', pady=10)
        
        # Main content
        content = tk.Frame(page, bg=BG_DEEP)
        content.pack(fill='both', expand=True, padx=40)
        
        # Left: algo selection
        left = tk.Frame(content, bg=BG_DEEP, width=340)
        left.pack(side='left', fill='y', padx=(0, 20))
        left.pack_propagate(False)
        
        tk.Label(left, text='◈  ALGORITHM', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.algo_cards = {}
        for name, desc in ALGO_DESC.items():
            card = AlgoCard(left, name, desc, ALGO_COLORS[name],
                          on_select=self._select_algo, width=310, height=90)
            card.pack(fill='x', pady=5)
            self.algo_cards[name] = card
        
        # Settings
        tk.Label(left, text='◈  SETTINGS', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(20, 10))
        
        settings = tk.Frame(left, bg=BG_CARD, padx=12, pady=10,
                           highlightthickness=1, highlightbackground=BORDER)
        settings.pack(fill='x')
        
        tk.Label(settings, text='Grid Size:', fg=TEXT_MID, bg=BG_CARD,
                font=('Courier New', 9)).grid(row=0, column=0, sticky='w', pady=4)
        self.grid_size_var = tk.StringVar(value='8')
        size_combo = ttk.Combobox(settings, textvariable=self.grid_size_var,
                                  values=['6', '8', '10', '12'], width=6, state='readonly')
        size_combo.grid(row=0, column=1, padx=8, sticky='w')
        
        tk.Label(settings, text='Episodes:', fg=TEXT_MID, bg=BG_CARD,
                font=('Courier New', 9)).grid(row=1, column=0, sticky='w', pady=4)
        self.episodes_var = tk.StringVar(value='500')
        ep_combo = ttk.Combobox(settings, textvariable=self.episodes_var,
                                values=['200', '500', '1000', '2000'], width=6, state='readonly')
        ep_combo.grid(row=1, column=1, padx=8, sticky='w')
        
        # Style combo boxes
        try:
            style = ttk.Style()
            style.configure('TCombobox', fieldbackground=BG_CARD, background=BG_CARD,
                            foreground=TEXT_HI, selectbackground=BORDER_HI)
        except Exception:
            pass
        
        # Right: info/preview panel
        right = tk.Frame(content, bg=BG_DEEP)
        right.pack(side='left', fill='both', expand=True)
        
        tk.Label(right, text='◈  ALGO INFO', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.info_panel = tk.Frame(right, bg=BG_CARD, padx=20, pady=16,
                                  highlightthickness=1, highlightbackground=BORDER)
        self.info_panel.pack(fill='x')
        
        self.info_text = tk.Text(self.info_panel, bg=BG_CARD, fg=TEXT_MID,
                                font=('Courier New', 10), height=10,
                                relief='flat', wrap='word', state='disabled',
                                insertbackground=GOLD)
        self.info_text.pack(fill='both', expand=True)
        self._update_info_panel(None)
        
        # Training progress section
        train_frame = tk.Frame(right, bg=BG_DEEP, pady=16)
        train_frame.pack(fill='x')
        
        tk.Label(train_frame, text='◈  TRAINING PROGRESS', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(0, 8))
        
        self.progress_bar = ProgressBar(train_frame, width=500, height=12, color=GOLD)
        self.progress_bar.pack(anchor='w')
        
        self.progress_label = tk.Label(train_frame, text='Select an algorithm to begin training.',
                                       fg=TEXT_DIM, bg=BG_DEEP, font=('Courier New', 9))
        self.progress_label.pack(anchor='w', pady=4)
        
        # Buttons
        btn_row = tk.Frame(right, bg=BG_DEEP)
        btn_row.pack(anchor='w', pady=10)
        
        self.train_btn = DKButton(btn_row, '⚡  TRAIN & PLAY',
                                 command=self._start_training,
                                 color=GOLD, width=200, height=44)
        self.train_btn.pack(side='left', padx=(0, 12))
        
        self.run_all_btn = DKButton(btn_row, '◉  TRAIN ALL',
                                   command=self._train_all_algos,
                                   color=BLUE, width=180, height=44)
        self.run_all_btn.pack(side='left')
    
    def _build_game_page(self):
        page = tk.Frame(self.page_frame, bg=BG_DEEP)
        self.pages['game'] = page
        
        # Header
        hdr = tk.Frame(page, bg=BG_DEEP)
        hdr.pack(fill='x', padx=40, pady=(24, 8))
        
        self.game_title_lbl = tk.Label(hdr, text='GRID WORLD — VALUE ITERATION',
                                       fg=GOLD, bg=BG_DEEP,
                                       font=('Courier New', 18, 'bold'))
        self.game_title_lbl.pack(side='left')
        
        tk.Frame(hdr, bg=GOLD_DIM, height=1).pack(fill='x', anchor='s', pady=(8, 0))
        
        # Main content
        content = tk.Frame(page, bg=BG_DEEP)
        content.pack(fill='both', expand=True, padx=40, pady=8)
        
        # Left: game canvas placeholder
        left = tk.Frame(content, bg=BG_DEEP)
        left.pack(side='left', fill='y', padx=(0, 24))
        
        self.game_canvas_frame = tk.Frame(left, bg=BG_BASE)
        self.game_canvas_frame.pack()
        
        self.game_canvas_placeholder = tk.Label(
            self.game_canvas_frame,
            text='Game will appear\nafter training.',
            fg=TEXT_DIM, bg=BG_BASE,
            font=('Courier New', 12),
            width=28, height=14
        )
        self.game_canvas_placeholder.pack(padx=4, pady=4)
        
        # Right: stats and controls
        right = tk.Frame(content, bg=BG_DEEP, width=320)
        right.pack(side='left', fill='y')
        right.pack_propagate(False)
        
        # Stats panel
        tk.Label(right, text='◈  GAME STATS', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(0, 8))
        
        stats_card = tk.Frame(right, bg=BG_CARD, padx=16, pady=12,
                             highlightthickness=1, highlightbackground=BORDER)
        stats_card.pack(fill='x')
        
        self.stat_labels = {}
        stat_defs = [
            ('steps', 'Steps', '0'),
            ('reward', 'Total Reward', '0.0'),
            ('pos', 'Position', '(0,0)'),
            ('status', 'Status', 'ACTIVE'),
        ]
        for key, label, init in stat_defs:
            row = tk.Frame(stats_card, bg=BG_CARD)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=f'{label}:', fg=TEXT_DIM, bg=BG_CARD,
                    font=('Courier New', 9), width=14, anchor='w').pack(side='left')
            lbl = tk.Label(row, text=init, fg=GOLD, bg=BG_CARD,
                          font=('Courier New', 9, 'bold'))
            lbl.pack(side='left')
            self.stat_labels[key] = lbl
        
        # Controls
        tk.Label(right, text='◈  CONTROLS', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(16, 8))
        
        ctrl = tk.Frame(right, bg=BG_CARD, padx=16, pady=12,
                       highlightthickness=1, highlightbackground=BORDER)
        ctrl.pack(fill='x')
        
        controls_text = [
            ('W / ↑', 'Move Up'),
            ('S / ↓', 'Move Down'),
            ('A / ←', 'Move Left'),
            ('D / →', 'Move Right'),
            ('Gold Arrow', 'AI Suggestion'),
            ('★', 'Goal Cell'),
        ]
        for key, desc in controls_text:
            row = tk.Frame(ctrl, bg=BG_CARD)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=key, fg=GOLD, bg=BG_CARD,
                    font=('Courier New', 9, 'bold'), width=12, anchor='w').pack(side='left')
            tk.Label(row, text=desc, fg=TEXT_DIM, bg=BG_CARD,
                    font=('Courier New', 9)).pack(side='left')
        
        # Game buttons
        tk.Label(right, text='◈  ACTIONS', fg=TEXT_MID, bg=BG_DEEP,
                font=('Courier New', 10, 'bold')).pack(anchor='w', pady=(16, 8))
        
        self.reset_btn = DKButton(right, '↺  RESET', command=self._reset_game,
                                 color=BLUE, width=280, height=38, font_size=10)
        self.reset_btn.pack(pady=4)
        
        self.autoplay_btn = DKButton(right, '▶  AI AUTO-PLAY', command=self._auto_play_game,
                                    color=CRIMSON, width=280, height=38, font_size=10)
        self.autoplay_btn.pack(pady=4)
        
        self.compare_btn = DKButton(right, '◎  COMPARE ALL', 
                                   command=self._run_analysis_and_show,
                                   color=GOLD, width=280, height=38, font_size=10)
        self.compare_btn.pack(pady=4)
        
        self.active_game_canvas = None
    
    def _build_analysis_page(self):
        page = tk.Frame(self.page_frame, bg=BG_DEEP)
        self.pages['analysis'] = page
        
        # Header
        hdr = tk.Frame(page, bg=BG_DEEP)
        hdr.pack(fill='x', padx=40, pady=(24, 8))
        
        tk.Label(hdr, text='COMPARATIVE ANALYSIS', fg=GOLD, bg=BG_DEEP,
                font=('Courier New', 20, 'bold')).pack(side='left')
        
        self.analysis_train_btn = DKButton(hdr, '⚡  TRAIN ALL & ANALYZE',
                                          command=self._train_all_algos,
                                          color=GOLD, width=220, height=36)
        self.analysis_train_btn.pack(side='right')
        
        tk.Frame(page, bg=GOLD_DIM, height=1).pack(fill='x', padx=40)
        
        # Analysis tabs
        tabs_frame = tk.Frame(page, bg=BG_DEEP)
        tabs_frame.pack(fill='x', padx=40, pady=8)
        
        self.analysis_tabs = {}
        tab_names = ['Learning Curves', 'Value Maps', 'Policy Grids', 'Radar Chart', 'Stats Table']
        
        for name in tab_names:
            btn = tk.Label(tabs_frame, text=name, fg=TEXT_DIM, bg=BG_DEEP,
                          font=('Courier New', 10, 'bold'), cursor='hand2',
                          padx=16, pady=6)
            btn.pack(side='left')
            btn.bind('<Enter>', lambda e, b=btn: b.configure(fg=GOLD) 
                     if b.cget('fg') == TEXT_DIM else None)
            btn.bind('<Leave>', lambda e, b=btn: b.configure(fg=TEXT_DIM)
                     if b.cget('fg') == GOLD else None)
            btn.bind('<Button-1>', lambda e, n=name: self._show_analysis_tab(n))
            self.analysis_tabs[name] = btn
        
        # Active tab indicator
        self.active_tab_indicator = tk.Frame(page, bg=GOLD, height=2, width=120)
        
        # Content area for analysis
        self.analysis_content = tk.Frame(page, bg=BG_DEEP)
        self.analysis_content.pack(fill='both', expand=True, padx=40, pady=4)
        
        self.analysis_placeholder = tk.Label(
            self.analysis_content,
            text='◈  Train all algorithms to generate comparative analysis.\n\nClick "TRAIN ALL & ANALYZE" to begin.',
            fg=TEXT_DIM, bg=BG_DEEP, font=('Courier New', 12), justify='center'
        )
        self.analysis_placeholder.pack(expand=True)
        
        self.analysis_image_label = None
        self.current_analysis_tab = None
        self.analysis_images = {}
    
    def _build_about_page(self):
        page = tk.Frame(self.page_frame, bg=BG_DEEP)
        self.pages['about'] = page
        
        content = tk.Frame(page, bg=BG_DEEP)
        content.pack(fill='both', expand=True, padx=60, pady=40)
        
        tk.Label(content, text='ABOUT THIS PROJECT', fg=GOLD, bg=BG_DEEP,
                font=('Courier New', 20, 'bold')).pack(anchor='w')
        tk.Frame(content, bg=GOLD_DIM, height=1).pack(fill='x', pady=12)
        
        about_sections = [
            ('◈  PROJECT TITLE', 
             'Comparative Analysis of Planning and Learning Approaches\nin Reinforcement Learning Using Grid World Environments'),
            ('◈  ALGORITHMS COMPARED',
             '1. Value Iteration (Model-Based Planning)\n   — Uses Bellman optimality equations with full MDP knowledge\n   — Guaranteed convergence to optimal policy\n   — Time complexity: O(|S|² × |A|) per iteration\n\n2. Q-Learning (Off-Policy TD Control)\n   — Learns Q(s,a) directly from experience\n   — Uses max future Q-value (off-policy)\n   — Converges to optimal regardless of behavior policy\n\n3. SARSA (On-Policy TD Control)\n   — Learns Q(s,a) using actual taken action\n   — More conservative, penalizes risky paths\n   — Better for environments with cliff-like hazards'),
            ('◈  ENVIRONMENT',
             'GridWorld — Configurable N×N grid with:\n  • Start state (top-left)\n  • Goal state (bottom-right)\n  • Random obstacle placement (15% density)\n  • Rewards: +100 (goal), -1 (step), -10 (timeout)'),
            ('◈  KEY METRICS',
             '• Convergence speed (iterations to θ-convergence)\n• Cumulative reward over training\n• Steps per episode\n• Computation time\n• Policy quality (optimal path length)'),
        ]
        
        for title, body in about_sections:
            tk.Label(content, text=title, fg=GOLD, bg=BG_DEEP,
                    font=('Courier New', 11, 'bold')).pack(anchor='w', pady=(12, 4))
            tk.Label(content, text=body, fg=TEXT_MID, bg=BG_DEEP,
                    font=('Courier New', 9), justify='left').pack(anchor='w', padx=16)
    
    # ── Page Navigation ───────────────────────────────────────────
    
    def _show_page(self, name):
        for p in self.pages.values():
            p.pack_forget()
        
        if name in self.pages:
            self.pages[name].pack(fill='both', expand=True)
            self.current_page = name
            
            # Update nav highlight
            for pn, btn in self.nav_buttons.items():
                btn.configure(fg=GOLD if pn == name else TEXT_DIM)
    
    # ── Algorithm Selection ────────────────────────────────────────
    
    def _select_algo(self, name):
        self.selected_algo = name
        for n, card in self.algo_cards.items():
            card.set_selected(n == name)
        self._update_info_panel(name)
    
    def _update_info_panel(self, name):
        info_texts = {
            'Value Iteration': """VALUE ITERATION — Model-Based Planning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Type: Dynamic Programming (Planning)
Approach: Requires full MDP knowledge
Update: V(s) ← max_a Σ P(s'|s,a)[R + γV(s')]

Characteristics:
  • Sweeps all states each iteration
  • Guaranteed optimal policy
  • No online learning — batch only
  • Converges in finite iterations for
    deterministic MDPs

Best for: When full model is available,
requires optimal solution guarantee""",

            'Q-Learning': """Q-LEARNING — Off-Policy TD Control
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Type: Model-Free Temporal Difference
Approach: Learn from experience (no model)
Update: Q(s,a) += α[r + γ max Q(s',·) - Q(s,a)]

Characteristics:
  • Off-policy: uses greedy max for target
  • Exploration via ε-greedy policy
  • Converges to optimal Q* 
  • Can learn "risky" but optimal paths

Best for: Unknown environments, when
optimal policy matters more than safety""",

            'SARSA': """SARSA — On-Policy TD Control
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Type: Model-Free Temporal Difference
Approach: Learn from actual experience
Update: Q(s,a) += α[r + γQ(s',a') - Q(s,a)]

Characteristics:
  • On-policy: uses actual next action
  • Learns policy agent is following
  • More conservative than Q-learning
  • Avoids risky regions during training

Best for: Safety-critical environments,
training with live consequences"""
        }
        
        self.info_text.configure(state='normal')
        self.info_text.delete('1.0', 'end')
        if name and name in info_texts:
            self.info_text.insert('end', info_texts[name])
        else:
            self.info_text.insert('end', 'Select an algorithm to see details.')
        self.info_text.configure(state='disabled')
    
    # ── Training ───────────────────────────────────────────────────
    
    def _start_training(self):
        if not self.selected_algo:
            self.progress_label.configure(text='⚠ Please select an algorithm first!', fg=WARNING)
            return
        
        algo = self.selected_algo
        self.progress_label.configure(text=f'Training {algo}...', fg=TEXT_MID)
        self.progress_bar.set(0)
        
        def run():
            from algorithms import GridWorld, ValueIteration, QLearning, SARSA
            
            size = int(self.grid_size_var.get())
            episodes = int(self.episodes_var.get())
            
            if self.env is None or self.env.size != size:
                self.env = GridWorld(size=size)
            
            color = ALGO_COLORS[algo]
            self.progress_bar._color = color
            
            def progress_cb(i, delta):
                pct = i / episodes if algo != 'Value Iteration' else min(i / 200, 1.0)
                self.after(0, lambda: self.progress_bar.set(pct))
                self.after(0, lambda: self.progress_label.configure(
                    text=f'Training {algo}... iter {i} | Δ={delta:.4f}', fg=TEXT_MID))
            
            if algo == 'Value Iteration':
                agent = ValueIteration(self.env)
            elif algo == 'Q-Learning':
                agent = QLearning(self.env, episodes=episodes)
            else:
                agent = SARSA(self.env, episodes=episodes)
            
            metrics = agent.train(callback=progress_cb)
            
            self.agents[algo] = agent
            self.metrics[algo] = metrics
            
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.progress_label.configure(
                text=f'✓ {algo} training complete! Time: {metrics["time_taken"]:.2f}s', fg=SUCCESS))
            self.after(0, lambda: self._launch_game(algo, agent))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _train_all_algos(self):
        self.progress_label.configure(text='Training all algorithms...', fg=TEXT_MID)
        self.progress_bar.set(0)
        
        def run():
            from algorithms import GridWorld, ValueIteration, QLearning, SARSA
            
            size = int(self.grid_size_var.get())
            episodes = int(self.episodes_var.get())
            
            self.env = GridWorld(size=size)
            
            algos = [
                ('Value Iteration', ValueIteration(self.env)),
                ('Q-Learning', QLearning(self.env, episodes=episodes)),
                ('SARSA', SARSA(self.env, episodes=episodes)),
            ]
            
            for i, (name, agent) in enumerate(algos):
                self.after(0, lambda n=name: self.progress_label.configure(
                    text=f'Training {n}...', fg=TEXT_MID))
                
                def cb(it, d, base=i, total=len(algos), ep=episodes):
                    pct = base/total + (it / ep) / total
                    self.after(0, lambda p=pct: self.progress_bar.set(p))
                
                metrics = agent.train(callback=cb)
                self.agents[name] = agent
                self.metrics[name] = metrics
            
            self.training_done = True
            
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.progress_label.configure(
                text='✓ All algorithms trained! Generating analysis...', fg=SUCCESS))
            self.after(0, self._generate_and_show_analysis)
        
        threading.Thread(target=run, daemon=True).start()
    
    def _run_analysis_and_show(self):
        if not self.training_done or len(self.agents) < 3:
            self._train_all_algos()
        else:
            self._generate_and_show_analysis()
    
    # ── Game Page ──────────────────────────────────────────────────
    
    def _launch_game(self, algo_name, agent):
        """Launch the game page with the trained agent"""
        self.game_title_lbl.configure(text=f'GRID WORLD — {algo_name.upper()}',
                                      fg=ALGO_COLORS[algo_name])
        
        # Clear old canvas
        for w in self.game_canvas_frame.winfo_children():
            w.destroy()
        
        def on_step(steps, reward, pos):
            self.stat_labels['steps'].configure(text=str(steps))
            self.stat_labels['reward'].configure(text=f'{reward:.1f}')
            self.stat_labels['pos'].configure(text=str(pos))
            if pos == self.env.goal:
                self.stat_labels['status'].configure(text='GOAL! ★', fg=GOLD)
        
        canvas = GridGameCanvas(self.game_canvas_frame, self.env, agent, 
                               algo_name, on_step_callback=on_step)
        canvas.pack(padx=4, pady=4)
        self.active_game_canvas = canvas
        
        # Reset stats
        self.stat_labels['steps'].configure(text='0')
        self.stat_labels['reward'].configure(text='0.0')
        self.stat_labels['pos'].configure(text=str(self.env.start))
        self.stat_labels['status'].configure(text='ACTIVE', fg=SUCCESS)
        
        self._show_page('game')
        canvas.focus_set()
    
    def _reset_game(self):
        if self.active_game_canvas:
            self.active_game_canvas.reset_game()
            self.stat_labels['steps'].configure(text='0')
            self.stat_labels['reward'].configure(text='0.0')
            self.stat_labels['status'].configure(text='ACTIVE', fg=SUCCESS)
            self.active_game_canvas.focus_set()
    
    def _auto_play_game(self):
        if self.active_game_canvas:
            if self.active_game_canvas.game_over:
                self.active_game_canvas.reset_game()
            self.active_game_canvas.auto_play()
    
    # ── Analysis Page ──────────────────────────────────────────────
    
    def _generate_and_show_analysis(self):
        if len(self.agents) < 3:
            return
        
        def run():
            from analysis import (plot_learning_curves, plot_value_heatmaps,
                                   plot_policy_grid, plot_radar_chart)
            from algorithms import get_comparison_stats
            
            self.after(0, lambda: self.analysis_placeholder.configure(
                text='Generating plots...'))
            
            bufs = {}
            bufs['Learning Curves'] = plot_learning_curves(self.metrics)
            bufs['Value Maps'] = plot_value_heatmaps(self.env, self.agents)
            bufs['Radar Chart'] = plot_radar_chart(get_comparison_stats(self.metrics))
            
            # Policy grids - combine into one image
            from PIL import Image
            policy_imgs = []
            for name, agent in self.agents.items():
                buf = plot_policy_grid(self.env, agent, name)
                policy_imgs.append(Image.open(buf))
            
            combined_w = sum(i.width for i in policy_imgs)
            combined_h = max(i.height for i in policy_imgs)
            combined = Image.new('RGB', (combined_w, combined_h))
            x = 0
            for img in policy_imgs:
                combined.paste(img, (x, 0))
                x += img.width
            
            from io import BytesIO
            cbuf = BytesIO()
            combined.save(cbuf, format='PNG')
            cbuf.seek(0)
            bufs['Policy Grids'] = cbuf
            
            # Stats table
            stats = get_comparison_stats(self.metrics)
            bufs['Stats Table'] = self._generate_stats_table(stats)
            
            self.analysis_images = bufs
            
            self.after(0, lambda: self._show_analysis_tab('Learning Curves'))
            self.after(0, lambda: self._show_page('analysis'))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _generate_stats_table(self, stats):
        """Generate a matplotlib stats table"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        fig.patch.set_facecolor('#08080e')
        
        # Table 1: Numeric stats
        ax = axes[0]
        ax.set_facecolor('#08080e')
        ax.axis('off')
        ax.set_title('Algorithm Comparison Statistics', color='#c9a84c', 
                     fontsize=14, fontweight='bold', pad=12)
        
        cols = ['Algorithm', 'Avg Reward\n(last 50)', 'Best Reward', 
                'Avg Steps', 'Best Steps', 'Train Time (s)', 'Convergence\nIter']
        rows = []
        for name, s in stats.items():
            rows.append([
                name,
                f'{s["avg_reward"]:.2f}',
                f'{s["best_reward"]:.1f}',
                f'{s["avg_steps"]:.1f}',
                f'{s["best_steps"]}',
                f'{s["time_taken"]:.3f}',
                f'{s["convergence"]}',
            ])
        
        colors_rows = [
            [ALGO_COLORS[r[0]] + '33'] * len(cols) for r in rows
        ]
        cell_colors = [['#12121f'] * len(cols)] * len(rows)
        
        table = ax.table(cellText=rows, colLabels=cols,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.2)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#12121f' if row > 0 else '#1a1a2a')
            cell.set_text_props(color='#e8e8f0' if row > 0 else '#c9a84c',
                               fontfamily='monospace')
            cell.set_edgecolor('#252538')
        
        # Table 2: Qualitative comparison
        ax2 = axes[1]
        ax2.set_facecolor('#08080e')
        ax2.axis('off')
        ax2.set_title('Qualitative Comparison', color='#c9a84c',
                      fontsize=14, fontweight='bold', pad=12)
        
        q_cols = ['Algorithm', 'Type', 'Needs Model', 'Policy Type', 'Optimality', 'Safety']
        q_rows = [
            ['Value Iteration', 'Dynamic Prog.', 'YES', 'Deterministic', 'Optimal', 'High'],
            ['Q-Learning',      'TD Off-Policy', 'NO',  'Stochastic(ε)', 'Optimal*', 'Medium'],
            ['SARSA',           'TD On-Policy',  'NO',  'Stochastic(ε)', 'Suboptimal*', 'High'],
        ]
        
        table2 = ax2.table(cellText=q_rows, colLabels=q_cols,
                          loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 2.2)
        
        for (row, col), cell in table2.get_celld().items():
            cell.set_facecolor('#12121f' if row > 0 else '#1a1a2a')
            cell.set_text_props(color='#e8e8f0' if row > 0 else '#c9a84c',
                               fontfamily='monospace')
            cell.set_edgecolor('#252538')
        
        plt.tight_layout(pad=2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                   facecolor='#08080e', edgecolor='none')
        plt.close()
        buf.seek(0)
        return buf
    
    def _show_analysis_tab(self, tab_name):
        if tab_name not in self.analysis_images:
            return
        
        self.current_analysis_tab = tab_name
        
        # Highlight active tab
        for name, btn in self.analysis_tabs.items():
            btn.configure(fg=GOLD if name == tab_name else TEXT_DIM)
        
        # Clear content
        for w in self.analysis_content.winfo_children():
            w.pack_forget()
        
        # Get image
        buf = self.analysis_images[tab_name]
        buf.seek(0)
        img = Image.open(buf)
        
        # Fit to window
        content_w = self.winfo_width() - 100
        content_h = self.winfo_height() - 200
        
        if content_w < 100:
            content_w = 1200
        if content_h < 100:
            content_h = 580
        
        aspect = img.width / img.height
        if content_w / content_h > aspect:
            new_h = min(content_h, img.height)
            new_w = int(new_h * aspect)
        else:
            new_w = min(content_w, img.width)
            new_h = int(new_w / aspect)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Convert to PhotoImage
        self._current_photo = ImageTk.PhotoImage(img_resized)
        
        lbl = tk.Label(self.analysis_content, image=self._current_photo,
                      bg=BG_DEEP)
        lbl.pack(expand=True)


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow not found. Installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
        from PIL import Image, ImageTk
    
    try:
        app = DarkKnightRL()
        app.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
