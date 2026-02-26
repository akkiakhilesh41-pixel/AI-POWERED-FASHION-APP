import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Upload, 
  Sparkles, 
  ShoppingBag, 
  MessageSquare, 
  History, 
  Info, 
  ChevronRight, 
  Camera, 
  Check, 
  Loader2,
  ArrowRight,
  ExternalLink,
  Menu,
  X,
  User,
  Heart
} from 'lucide-react';
import Markdown from 'react-markdown';
import { analyzeFashionItem, getStylistResponse, FashionAnalysis, Recommendation } from './services/geminiService';

// --- Types ---
type Page = 'home' | 'upload' | 'processing' | 'results' | 'chat' | 'wardrobe' | 'about';

interface WardrobeItem {
  id: number;
  image_data: string;
  analysis: FashionAnalysis;
  recommendations: Recommendation[];
  created_at: string;
}

// --- Components ---

const Navbar = ({ currentPage, setPage }: { currentPage: Page, setPage: (p: Page) => void }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const navItems: { id: Page, label: string, icon: any }[] = [
    { id: 'home', label: 'Home', icon: Sparkles },
    { id: 'upload', label: 'Upload Outfit', icon: Upload },
    { id: 'chat', label: 'Stylist Chat', icon: MessageSquare },
    { id: 'wardrobe', label: 'Wardrobe', icon: History },
    { id: 'about', label: 'About', icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md border-b border-white/10">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        <div 
          className="flex items-center gap-2 cursor-pointer" 
          onClick={() => setPage('home')}
        >
          <div className="w-10 h-10 bg-white flex items-center justify-center rounded-full">
            <span className="text-black font-bold text-xl">L</span>
          </div>
          <span className="text-white font-display text-2xl tracking-tighter font-bold uppercase">Lumière</span>
        </div>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={`text-sm uppercase tracking-widest transition-colors ${
                currentPage === item.id ? 'text-white font-semibold' : 'text-white/50 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
          <button 
            onClick={() => setPage('upload')}
            className="bg-white text-black px-6 py-2 rounded-full text-xs font-bold uppercase tracking-widest hover:bg-beige-100 transition-colors"
          >
            Get Styled
          </button>
        </div>

        {/* Mobile Menu Toggle */}
        <button className="md:hidden text-white" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Nav */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden bg-black border-b border-white/10 px-6 py-8 flex flex-col gap-6"
          >
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => { setPage(item.id); setIsOpen(false); }}
                className={`text-left text-lg uppercase tracking-widest ${
                  currentPage === item.id ? 'text-white' : 'text-white/50'
                }`}
              >
                {item.label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

// --- Pages ---

const HomePage = ({ setPage }: { setPage: (p: Page) => void }) => {
  return (
    <div className="pt-20">
      {/* Hero Section */}
      <section className="relative h-[90vh] flex items-center overflow-hidden">
        <div className="absolute inset-0 z-0">
          <img 
            src="https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&q=80&w=2070" 
            className="w-full h-full object-cover opacity-40"
            alt="Fashion background"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-black via-black/60 to-transparent" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-6 w-full">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-2xl"
          >
            <div className="flex items-center gap-2 mb-6">
              <div className="h-[1px] w-12 bg-white/50" />
              <span className="text-white/70 uppercase tracking-[0.3em] text-xs font-semibold">AI Fashion Intelligence</span>
            </div>
            <h1 className="text-6xl md:text-8xl font-bold text-white leading-tight mb-8 tracking-tighter">
              Upload Your Outfit.<br />
              <span className="text-white/40 italic">Let AI Style You.</span>
            </h1>
            <p className="text-white/60 text-lg mb-10 max-w-lg leading-relaxed">
              Experience the future of personal styling. Our advanced neural networks analyze your garments to curate the perfect ensemble from global luxury platforms.
            </p>
            <div className="flex flex-wrap gap-4">
              <button 
                onClick={() => setPage('upload')}
                className="bg-white text-black px-10 py-5 rounded-full font-bold uppercase tracking-widest hover:scale-105 transition-transform flex items-center gap-2"
              >
                Upload Outfit <ArrowRight size={18} />
              </button>
              <button 
                onClick={() => setPage('about')}
                className="border border-white/20 text-white px-10 py-5 rounded-full font-bold uppercase tracking-widest hover:bg-white/10 transition-colors"
              >
                How it works
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="py-32 bg-[#0a0a0a]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold text-white mb-4">The Lumière Workflow</h2>
            <p className="text-white/50">Precision engineering meets high-fashion curation.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { icon: Upload, title: "Upload", desc: "Share a photo of any garment in your collection." },
              { icon: Sparkles, title: "Analyze", desc: "AI detects color, texture, and silhouette with 99% accuracy." },
              { icon: ShoppingBag, title: "Match", desc: "Our engine scans thousands of items to find your perfect match." },
              { icon: Check, title: "Acquire", desc: "Direct links to Myntra, Amazon, and Ajio for seamless shopping." }
            ].map((step, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-8 rounded-3xl bg-white/5 border border-white/10 hover:border-white/30 transition-colors group"
              >
                <div className="w-16 h-16 rounded-2xl bg-white/10 flex items-center justify-center mb-6 group-hover:bg-white group-hover:text-black transition-colors">
                  <step.icon size={28} />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
                <p className="text-white/40 text-sm leading-relaxed">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

const UploadPage = ({ onUpload }: { onUpload: (file: File) => void }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="pt-32 pb-20 min-h-screen flex items-center justify-center px-6">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 tracking-tight">Upload Your Item</h1>
          <p className="text-white/50">Drag and drop a photo of your clothing to begin the AI styling process.</p>
        </div>

        <motion.div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative aspect-video rounded-[40px] border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center gap-6 ${
            dragActive ? 'border-white bg-white/10' : 'border-white/20 bg-white/5 hover:bg-white/10'
          }`}
        >
          <input ref={inputRef} type="file" className="hidden" accept="image/*" onChange={handleChange} />
          
          <div className="w-24 h-24 rounded-full bg-white/10 flex items-center justify-center text-white/50">
            <Upload size={40} />
          </div>
          <div className="text-center">
            <p className="text-xl text-white font-medium mb-2">Drop your image here</p>
            <p className="text-white/40 text-sm">Supports JPG, PNG (Max 5MB)</p>
          </div>

          {/* Decorative AI scanning lines */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-[40px]">
            <motion.div 
              animate={{ y: ['0%', '100%', '0%'] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              className="w-full h-[2px] bg-white/20 shadow-[0_0_15px_rgba(255,255,255,0.5)]"
            />
          </div>
        </motion.div>

        <div className="mt-12 grid grid-cols-3 gap-6">
          {[
            "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&q=80&w=400",
            "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?auto=format&fit=crop&q=80&w=400",
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&q=80&w=400"
          ].map((img, i) => (
            <div key={i} className="rounded-2xl overflow-hidden aspect-square opacity-30 hover:opacity-100 transition-opacity cursor-pointer">
              <img src={img} className="w-full h-full object-cover" alt="Example" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const ProcessingPage = ({ image }: { image: string }) => {
  return (
    <div className="pt-20 min-h-screen flex flex-col items-center justify-center px-6">
      <div className="relative w-80 h-80 mb-12">
        <img src={image} className="w-full h-full object-cover rounded-3xl" alt="Processing" />
        <div className="absolute inset-0 bg-black/40 rounded-3xl" />
        
        {/* Scanning Animation */}
        <motion.div 
          animate={{ y: ['0%', '100%', '0%'] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="absolute left-0 right-0 h-1 bg-white shadow-[0_0_20px_white] z-10"
        />

        {/* Neural Network Dots */}
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            animate={{ 
              opacity: [0.2, 1, 0.2],
              scale: [1, 1.2, 1]
            }}
            transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
            className="absolute w-2 h-2 bg-white rounded-full"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
          />
        ))}
      </div>

      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
          <Loader2 className="animate-spin" /> Analyzing Your Style...
        </h2>
        <div className="flex gap-2 justify-center">
          {['Color Detection', 'Pattern Analysis', 'Style Mapping'].map((tag, i) => (
            <motion.span 
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.5 }}
              className="px-4 py-1 rounded-full bg-white/10 text-white/60 text-xs uppercase tracking-widest border border-white/5"
            >
              {tag}
            </motion.span>
          ))}
        </div>
      </div>
    </div>
  );
};

const ResultsPage = ({ image, analysis, recommendations }: { image: string, analysis: FashionAnalysis, recommendations: Recommendation[] }) => {
  return (
    <div className="pt-32 pb-20 max-w-7xl mx-auto px-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
        
        {/* Left: Original Item */}
        <div className="lg:col-span-4">
          <div className="sticky top-32">
            <h2 className="text-sm uppercase tracking-[0.3em] text-white/40 mb-6">Your Selection</h2>
            <div className="rounded-[40px] overflow-hidden bg-white/5 border border-white/10 p-4">
              <img src={image} className="w-full aspect-square object-cover rounded-[32px] mb-6" alt="Original" />
              <div className="space-y-4">
                <div>
                  <h3 className="text-white font-bold text-xl mb-1">{analysis.category}</h3>
                  <p className="text-white/50 text-sm">{analysis.description}</p>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/5 p-3 rounded-2xl border border-white/5">
                    <p className="text-[10px] uppercase text-white/30 mb-1">Color</p>
                    <p className="text-white text-sm font-medium">{analysis.color}</p>
                  </div>
                  <div className="bg-white/5 p-3 rounded-2xl border border-white/5">
                    <p className="text-[10px] uppercase text-white/30 mb-1">Style</p>
                    <p className="text-white text-sm font-medium">{analysis.style}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Recommendations */}
        <div className="lg:col-span-8">
          <h2 className="text-sm uppercase tracking-[0.3em] text-white/40 mb-6">Curated Matches</h2>
          <div className="space-y-8">
            {recommendations.map((item, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="group relative flex flex-col md:flex-row gap-8 p-6 rounded-[40px] bg-white/5 border border-white/10 hover:bg-white/10 transition-all"
              >
                <div className="w-full md:w-48 h-48 rounded-3xl overflow-hidden flex-shrink-0">
                  <img 
                    src={`https://picsum.photos/seed/${item.name}/400/400`} 
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" 
                    alt={item.name} 
                  />
                </div>
                <div className="flex-grow flex flex-col justify-between py-2">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] uppercase tracking-widest text-white/40">{item.category}</span>
                      <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-white/10 border border-white/10">
                        <Sparkles size={12} className="text-white" />
                        <span className="text-[10px] font-bold text-white">{item.matchScore}% Match</span>
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-2">{item.name}</h3>
                    <p className="text-white/50 text-sm mb-4 leading-relaxed">{item.reason}</p>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[10px] uppercase text-white/30">Available on</p>
                      <p className="text-white font-bold">{item.platform}</p>
                    </div>
                    <a 
                      href={item.purchaseUrl || "#"} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="bg-white text-black px-8 py-3 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-2 hover:bg-beige-100"
                    >
                      Buy Now <ExternalLink size={14} />
                    </a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const ChatPage = () => {
  const [messages, setMessages] = useState<{ role: 'user' | 'model', content: string, imageUrl?: string, hairImageUrl?: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const response = await getStylistResponse(userMsg, messages);
      setMessages(prev => [...prev, { 
        role: 'model', 
        content: response.text || "Here is a visual representation of the styling idea.",
        imageUrl: response.imageUrl,
        hairImageUrl: response.hairImageUrl
      }]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pt-20 h-screen flex flex-col bg-black">
      <div className="max-w-5xl mx-auto w-full flex-grow flex flex-col p-6 overflow-hidden">
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center">
            <User className="text-black" />
          </div>
          <div>
            <h2 className="text-white font-bold text-xl">Lumière Stylist</h2>
            <p className="text-white/40 text-xs uppercase tracking-widest">AI Visual Consultant</p>
          </div>
        </div>

        <div ref={scrollRef} className="flex-grow overflow-y-auto space-y-6 mb-6 scrollbar-hide">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center px-12">
              <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mb-6">
                <Sparkles className="text-white/20" size={32} />
              </div>
              <h3 className="text-white text-xl font-bold mb-2">Visual Styling Chat</h3>
              <p className="text-white/40 text-sm">"Show me a casual summer outfit for a beach party."<br />"What would look good with a navy blazer?"</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] p-6 rounded-[32px] ${
                msg.role === 'user' 
                  ? 'bg-white text-black rounded-tr-none' 
                  : 'bg-white/5 text-white border border-white/10 rounded-tl-none'
              }`}>
                {(msg.imageUrl || msg.hairImageUrl) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    {msg.imageUrl && (
                      <div className="rounded-2xl overflow-hidden border border-white/10">
                        <p className="text-[10px] uppercase tracking-widest text-white/40 p-2 bg-white/5 border-b border-white/10">Full Outfit</p>
                        <img src={msg.imageUrl} alt="Styling Idea" className="w-full h-auto object-cover" />
                      </div>
                    )}
                    {msg.hairImageUrl && (
                      <div className="rounded-2xl overflow-hidden border border-white/10">
                        <p className="text-[10px] uppercase tracking-widest text-white/40 p-2 bg-white/5 border-b border-white/10">Hairstyle Idea</p>
                        <img src={msg.hairImageUrl} alt="Hairstyle Idea" className="w-full h-auto object-cover" />
                      </div>
                    )}
                  </div>
                )}
                <div className="markdown-body text-sm leading-relaxed prose prose-invert max-w-none">
                  <Markdown>{msg.content}</Markdown>
                </div>
              </div>
            </motion.div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white/5 p-6 rounded-[32px] rounded-tl-none border border-white/10">
                <div className="flex flex-col gap-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="w-32 h-40 bg-white/5 animate-pulse rounded-2xl flex items-center justify-center">
                      <Camera className="text-white/10" size={24} />
                    </div>
                    <div className="w-32 h-32 bg-white/5 animate-pulse rounded-2xl flex items-center justify-center">
                      <Camera className="text-white/10" size={24} />
                    </div>
                  </div>
                  <Loader2 className="animate-spin text-white/40" />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="relative">
          <input 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Describe an outfit you want to see..."
            className="w-full bg-white/5 border border-white/10 rounded-full py-5 px-8 text-white focus:outline-none focus:border-white/30 transition-colors"
          />
          <button 
            onClick={handleSend}
            className="absolute right-3 top-1/2 -translate-y-1/2 w-12 h-12 bg-white rounded-full flex items-center justify-center text-black hover:scale-105 transition-transform"
          >
            <ArrowRight size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

const WardrobePage = () => {
  const [items, setItems] = useState<WardrobeItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/wardrobe')
      .then(res => res.json())
      .then(data => {
        setItems(data);
        setLoading(false);
      });
  }, []);

  return (
    <div className="pt-32 pb-20 max-w-7xl mx-auto px-6">
      <div className="flex items-center justify-between mb-12">
        <div>
          <h1 className="text-5xl font-bold text-white mb-2 tracking-tight">Digital Wardrobe</h1>
          <p className="text-white/50">Your curated collection of styles and recommendations.</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-white font-bold text-2xl">{items.length}</p>
            <p className="text-white/30 text-[10px] uppercase tracking-widest">Total Items</p>
          </div>
          <div className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
            <History className="text-white/40" />
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="animate-spin text-white/20" size={40} />
        </div>
      ) : items.length === 0 ? (
        <div className="text-center py-32 bg-white/5 rounded-[40px] border border-white/10">
          <ShoppingBag className="mx-auto text-white/10 mb-6" size={64} />
          <h3 className="text-white text-2xl font-bold mb-2">Your wardrobe is empty</h3>
          <p className="text-white/40 mb-8">Start by uploading your first outfit item.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {items.map((item) => (
            <motion.div 
              key={item.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="group relative rounded-[40px] overflow-hidden bg-white/5 border border-white/10 hover:border-white/30 transition-all"
            >
              <div className="aspect-square overflow-hidden">
                <img src={item.image_data} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" alt="Wardrobe Item" />
              </div>
              <div className="p-8">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] uppercase tracking-widest text-white/40">{item.analysis.category}</span>
                  <span className="text-[10px] text-white/30">{new Date(item.created_at).toLocaleDateString()}</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{item.analysis.style} {item.analysis.color}</h3>
                <p className="text-white/50 text-sm line-clamp-2 mb-6">{item.analysis.description}</p>
                <div className="flex items-center gap-2">
                  <div className="flex -space-x-2">
                    {item.recommendations.slice(0, 3).map((rec, i) => (
                      <div key={i} className="w-8 h-8 rounded-full border-2 border-black bg-white/20 overflow-hidden">
                        <img src={`https://picsum.photos/seed/${rec.name}/50/50`} className="w-full h-full object-cover" alt="Rec" />
                      </div>
                    ))}
                  </div>
                  <span className="text-[10px] text-white/40">+{item.recommendations.length} recommendations</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

const AboutPage = () => {
  return (
    <div className="pt-32 pb-20 max-w-7xl mx-auto px-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center mb-32">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
        >
          <div className="flex items-center gap-2 mb-6">
            <div className="h-[1px] w-12 bg-white/50" />
            <span className="text-white/70 uppercase tracking-[0.3em] text-xs font-semibold">Our Vision</span>
          </div>
          <h1 className="text-6xl font-bold text-white leading-tight mb-8 tracking-tighter">
            Where Intelligence<br />
            <span className="text-white/40 italic">Meets Elegance.</span>
          </h1>
          <p className="text-white/60 text-lg leading-relaxed mb-8">
            Lumière was born from a simple observation: the digital shopping experience lacks the nuance of a personal stylist. We've bridged that gap using state-of-the-art Computer Vision and Generative AI.
          </p>
          <div className="space-y-6">
            {[
              { title: "Computer Vision", desc: "Our models identify over 2,000 unique fashion attributes from a single image." },
              { title: "Semantic Matching", desc: "We don't just match colors; we understand style archetypes and cultural trends." },
              { title: "Global Curation", desc: "Access a unified feed from Amazon, Myntra, and Ajio in real-time." }
            ].map((feature, i) => (
              <div key={i} className="flex gap-4">
                <div className="w-1 h-12 bg-white/20 rounded-full flex-shrink-0" />
                <div>
                  <h4 className="text-white font-bold uppercase tracking-widest text-xs mb-1">{feature.title}</h4>
                  <p className="text-white/40 text-sm">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="relative"
        >
          <div className="aspect-[4/5] rounded-[60px] overflow-hidden">
            <img 
              src="https://images.unsplash.com/photo-1539109132314-34a77ae68c44?auto=format&fit=crop&q=80&w=1000" 
              className="w-full h-full object-cover"
              alt="Fashion Editorial"
            />
          </div>
          <div className="absolute -bottom-10 -right-10 w-64 h-64 bg-white p-8 rounded-[40px] hidden md:block">
            <p className="text-black font-display text-4xl font-bold leading-none mb-4">99%</p>
            <p className="text-black/60 text-sm uppercase tracking-widest font-bold">Accuracy in Style Detection</p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<FashionAnalysis | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const handleUpload = async (file: File) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64 = e.target?.result as string;
      setUploadedImage(base64);
      setPage('processing');
      
      try {
        const result = await analyzeFashionItem(base64);
        setAnalysis(result.analysis);
        setRecommendations(result.recommendations);
        
        // Save to wardrobe
        await fetch('/api/wardrobe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_data: base64,
            analysis: result.analysis,
            recommendations: result.recommendations
          })
        });

        setPage('results');
      } catch (err) {
        console.error("Analysis failed:", err);
        setPage('upload');
        alert("Styling analysis failed. Please try again.");
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-white selection:text-black">
      <Navbar currentPage={page} setPage={setPage} />
      
      <main>
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.4 }}
          >
            {page === 'home' && <HomePage setPage={setPage} />}
            {page === 'upload' && <UploadPage onUpload={handleUpload} />}
            {page === 'processing' && uploadedImage && <ProcessingPage image={uploadedImage} />}
            {page === 'results' && uploadedImage && analysis && (
              <ResultsPage 
                image={uploadedImage} 
                analysis={analysis} 
                recommendations={recommendations} 
              />
            )}
            {page === 'chat' && <ChatPage />}
            {page === 'wardrobe' && <WardrobePage />}
            {page === 'about' && <AboutPage />}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="py-20 border-t border-white/10 bg-black">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="col-span-2">
            <div className="flex items-center gap-2 mb-6">
              <div className="w-8 h-8 bg-white flex items-center justify-center rounded-full">
                <span className="text-black font-bold">L</span>
              </div>
              <span className="text-white font-display text-xl tracking-tighter font-bold uppercase">Lumière</span>
            </div>
            <p className="text-white/40 max-w-sm mb-8">
              Redefining the digital fashion landscape through artificial intelligence and curated luxury.
            </p>
            <div className="flex gap-4">
              {['Instagram', 'Twitter', 'Vogue'].map(social => (
                <a key={social} href="#" className="text-xs uppercase tracking-widest text-white/60 hover:text-white transition-colors">{social}</a>
              ))}
            </div>
          </div>
          <div>
            <h4 className="text-white font-bold uppercase tracking-widest text-xs mb-6">Navigation</h4>
            <ul className="space-y-4 text-white/40 text-sm">
              <li><button onClick={() => setPage('home')}>Home</button></li>
              <li><button onClick={() => setPage('upload')}>Upload Outfit</button></li>
              <li><button onClick={() => setPage('chat')}>Stylist Chat</button></li>
              <li><button onClick={() => setPage('wardrobe')}>Wardrobe</button></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-bold uppercase tracking-widest text-xs mb-6">Legal</h4>
            <ul className="space-y-4 text-white/40 text-sm">
              <li><a href="#">Privacy Policy</a></li>
              <li><a href="#">Terms of Service</a></li>
              <li><a href="#">Cookie Policy</a></li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-6 mt-20 pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-white/20 text-[10px] uppercase tracking-widest">© 2026 Lumière AI. All rights reserved.</p>
          <p className="text-white/20 text-[10px] uppercase tracking-widest">Designed for the future of fashion.</p>
        </div>
      </footer>
    </div>
  );
}
