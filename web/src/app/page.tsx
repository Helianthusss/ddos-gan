"use client";
import React, { useState } from 'react';
import axios from 'axios';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { Shield, ShieldAlert, Cpu, Activity, Play, Database, CheckCircle, Zap, Crosshair, Server } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type PacketData = {
    features: number[];
    type: string;
};

export default function Dashboard() {
    const [packet, setPacket] = useState<PacketData | null>(null);
    const [detecting, setDetecting] = useState(false);
    const [result, setResult] = useState<{probability: number, status: string} | null>(null);
    const [detectorVersion, setDetectorVersion] = useState("v1");

    const fetchRealData = async () => {
        setResult(null);
        setPacket(null);
        const res = await axios.get(`${API_BASE}/sample/real`);
        setPacket(res.data);
    };

    const generateGAN = async (round: number) => {
        setResult(null);
        setPacket(null);
        const res = await axios.post(`${API_BASE}/generate?round=${round}`);
        setPacket(res.data);
    };

    const runDetector = async () => {
        if (!packet) return;
        setDetecting(true);
        try {
            const res = await axios.post(`${API_BASE}/detect?model=${detectorVersion}`, { features: packet.features });
            setResult(res.data);
        } catch (e) {
            console.error(e);
        }
        setDetecting(false);
    };

    const chartData = packet ? packet.features.slice(0, 15).map((val, i) => ({
        feature: `F${i}`,
        value: val,
    })) : [];

    return (
        <div className="min-h-screen font-sans flex flex-col items-center p-4 lg:p-8 selection:bg-emerald-500/30">
            {/* NAVBAR */}
            <nav className="w-full max-w-[1400px] glass-panel rounded-2xl p-4 md:px-6 flex justify-between items-center mb-8">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-tr from-emerald-500 to-cyan-500 rounded-lg shadow-lg shadow-emerald-500/20">
                        <Shield className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-white tracking-wide">Nexus Sentinel</h1>
                        <p className="text-xs text-slate-400">Adversarial ML Threat Simulator</p>
                    </div>
                </div>
                <div className="flex gap-6 text-sm">
                    <div className="flex flex-col items-end">
                        <span className="text-slate-500 text-[10px] uppercase font-bold tracking-wider">System Status</span>
                        <span className="text-emerald-400 flex items-center gap-2 font-medium">
                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> ONLINE
                        </span>
                    </div>
                </div>
            </nav>

            {/* MAIN CONTENT */}
            <main className="w-full max-w-[1400px] flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
                
                {/* LEFT PANEL - ATTACKER */}
                <div className="lg:col-span-3 glass-panel rounded-2xl flex flex-col overflow-hidden relative">
                    <div className="px-6 py-5 border-b border-white/5 bg-white/[0.02]">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            <Crosshair className="w-5 h-5 text-rose-500" />
                            Threat Generator
                        </h2>
                    </div>
                    <div className="p-6 flex flex-col gap-4">
                        <p className="text-sm text-slate-400 mb-2 leading-relaxed">Init network traffic payload to test IDS resilience.</p>
                        
                        <button onClick={fetchRealData} className={`group flex py-4 px-5 rounded-xl border transition-all duration-300 text-left items-center ${packet?.type.includes('Real') ? 'bg-blue-500/10 border-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.15)]' : 'bg-slate-800/30 border-white/5 hover:border-slate-500 hover:bg-slate-800/50'}`}>
                            <Database className="w-6 h-6 text-blue-400 mr-4 shrink-0 group-hover:scale-110 transition-transform" />
                            <div>
                                <h3 className="font-bold text-slate-200">Real DDoS</h3>
                                <p className="text-xs text-slate-400 mt-0.5">Standard CIC-DDoS2019 botnet signature</p>
                            </div>
                        </button>

                        <button onClick={() => generateGAN(0)} className={`group flex py-4 px-5 rounded-xl border transition-all duration-300 text-left items-center ${packet?.type.includes('Round 0') ? 'bg-orange-500/10 border-orange-500/50 shadow-[0_0_20px_rgba(249,115,22,0.15)]' : 'bg-slate-800/30 border-white/5 hover:border-slate-500 hover:bg-slate-800/50'}`}>
                            <Cpu className="w-6 h-6 text-orange-400 mr-4 shrink-0 group-hover:scale-110 transition-transform" />
                            <div>
                                <h3 className="font-bold text-slate-200">Pure GAN (R0)</h3>
                                <p className="text-xs text-slate-400 mt-0.5">AI-generated payload without evasion constraints</p>
                            </div>
                        </button>

                        <button onClick={() => generateGAN(2)} className={`group flex py-4 px-5 rounded-xl border transition-all duration-300 text-left items-center ${packet?.type.includes('Round 2') ? 'bg-purple-500/10 border-purple-500/50 shadow-[0_0_20px_rgba(168,85,247,0.15)]' : 'bg-slate-800/30 border-white/5 hover:border-slate-500 hover:bg-slate-800/50'}`}>
                            <Zap className="w-6 h-6 text-purple-400 mr-4 shrink-0 group-hover:scale-110 transition-transform" />
                            <div>
                                <h3 className="font-bold text-slate-200">Adversarial GAN (R2)</h3>
                                <p className="text-xs text-slate-400 mt-0.5">Hardened AI payload tuned to bypass detection</p>
                            </div>
                        </button>
                    </div>
                </div>

                {/* MIDDLE PANEL - VISUALIZATION */}
                <div className="lg:col-span-6 glass-panel rounded-2xl flex flex-col overflow-hidden">
                    <div className="px-6 py-5 border-b border-white/5 bg-white/[0.02] flex justify-between items-center">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-400" />
                            Payload Telemetry
                        </h2>
                        {packet && (
                            <span className="text-xs font-mono bg-slate-900 border border-slate-700 px-3 py-1 rounded-full text-slate-400 shadow-inner">
                                Vector: 80-Dim Float
                            </span>
                        )}
                    </div>
                    
                    <div className="flex-1 p-6 flex flex-col justify-center items-center min-h-[450px]">
                        {packet ? (
                            <div className="w-full h-full flex flex-col items-center">
                                <h3 className="text-center font-bold tracking-widest uppercase mb-2 text-sm" 
                                    style={{ color: packet.type.includes('Real') ? '#60a5fa' : packet.type.includes('2') ? '#c084fc' : '#fb923c' }}>
                                    {packet.type} LOADED
                                </h3>
                                <p className="text-xs text-slate-500 mb-6 text-center">Interactive mapping of Top 15 network features</p>
                                
                                <div className="w-full flex-1 max-h-[360px] max-w-md">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
                                            <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                            <PolarAngleAxis dataKey="feature" tick={{ fill: '#64748b', fontSize: 11 }} />
                                            <Tooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.95)', borderColor: '#1e293b', color: '#fff', borderRadius: '12px', backdropFilter: 'blur(8px)' }} itemStyle={{ color: '#fff' }} />
                                            <Radar 
                                                name="Normalized Scope" 
                                                dataKey="value" 
                                                stroke={packet.type.includes('2') ? '#a855f7' : packet.type.includes('Real') ? '#3b82f6' : '#f97316'} 
                                                fill={packet.type.includes('2') ? '#a855f7' : packet.type.includes('Real') ? '#3b82f6' : '#f97316'} 
                                                strokeWidth={2}
                                                fillOpacity={0.3} 
                                            />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center text-slate-500/70">
                                <div className="relative mb-6">
                                    <div className="absolute inset-0 bg-slate-500/20 blur-xl rounded-full"></div>
                                    <Server className="w-20 h-20 relative z-10 opacity-50" />
                                </div>
                                <p className="text-sm tracking-wide uppercase font-semibold">Awaiting Payload Injection</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* RIGHT PANEL - IDS TARGET */}
                <div className="lg:col-span-3 glass-panel rounded-2xl flex flex-col overflow-hidden">
                    <div className="px-6 py-5 border-b border-white/5 bg-white/[0.02]">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            <Shield className="w-5 h-5 text-emerald-400" />
                            Target IDS Engine
                        </h2>
                    </div>
                    
                    <div className="p-6 flex flex-col h-full">
                        <div className="mb-6">
                            <label className="text-xs uppercase tracking-wider text-slate-500 mb-2 block font-semibold">Engine Configuration</label>
                            <div className="relative">
                                <select 
                                    value={detectorVersion} 
                                    onChange={(e) => setDetectorVersion(e.target.value)}
                                    className="w-full appearance-none bg-slate-900/80 border border-slate-700 hover:border-emerald-500/40 text-slate-200 py-3.5 pl-4 pr-10 rounded-xl outline-none cursor-pointer transition-colors text-sm font-medium shadow-inner focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                                >
                                    <option value="v1">v1 / Baseline MLP Filter</option>
                                    <option value="v2">v2 / Hardened MLP (Adv. Training)</option>
                                </select>
                                <div className="absolute inset-y-0 right-4 flex items-center pointer-events-none">
                                    <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                </div>
                            </div>
                        </div>

                        <button 
                            onClick={runDetector} 
                            disabled={!packet || detecting}
                            className={`w-full py-4 rounded-xl font-bold tracking-wide transition-all flex justify-center items-center gap-2 mb-8 ${detecting || !packet ? 'bg-slate-800 text-slate-500 cursor-not-allowed opacity-60' : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] transform hover:-translate-y-0.5'}`}
                        >
                            {detecting ? <Cpu className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5 fill-current" />}
                            {detecting ? "ANALYZING..." : "DEPLOY PAYLOAD"}
                        </button>

                        <div className="flex-1 bg-slate-950/60 rounded-xl border border-slate-800/80 p-6 flex flex-col justify-center items-center relative shadow-inner">
                            {!result && !detecting && (
                                <div className="text-slate-600 flex flex-col items-center">
                                    <span className="w-2 h-2 rounded-full bg-slate-700 mb-3 animate-pulse"></span>
                                    <span className="text-xs uppercase font-bold tracking-widest">Scanner Standby</span>
                                </div>
                            )}
                            {detecting && (
                                <div className="text-teal-400 text-sm flex flex-col items-center w-full">
                                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mb-5 relative shadow-inner">
                                        <div className="absolute top-0 h-full bg-teal-400 animate-[scan_1.2s_ease-in-out_infinite] w-1/2 drop-shadow-[0_0_5px_rgba(45,212,191,0.8)]"></div>
                                    </div>
                                    <span className="uppercase font-semibold tracking-wide">Inspecting Tensors...</span>
                                </div>
                            )}
                            {result && !detecting && (
                                <div className="w-full flex-1 flex flex-col justify-between animate-in fade-in zoom-in-95 duration-500">
                                    {result.status === 'BLOCKED' ? (
                                        <div className="flex flex-col items-center flex-1 justify-center">
                                            <div className="relative">
                                                <div className="absolute inset-0 bg-rose-500/30 blur-2xl rounded-full"></div>
                                                <ShieldAlert className="w-16 h-16 text-rose-500 relative z-10 mb-4 drop-shadow-[0_0_15px_rgba(244,63,94,0.5)]" />
                                            </div>
                                            <h3 className="text-3xl font-black text-rose-500 tracking-wider mb-2">BLOCKED</h3>
                                            <p className="text-rose-400/80 text-xs text-center px-2">Malicious signature identified</p>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center flex-1 justify-center">
                                            <div className="relative">
                                                <div className="absolute inset-0 bg-emerald-500/30 blur-2xl rounded-full"></div>
                                                <CheckCircle className="w-16 h-16 text-emerald-400 relative z-10 mb-4 drop-shadow-[0_0_15px_rgba(52,211,153,0.5)]" />
                                            </div>
                                            <h3 className="text-3xl font-black text-emerald-400 tracking-wider mb-2">SAFE</h3>
                                            <p className="text-emerald-400/80 text-xs text-center px-2">Traffic payload passed undetected</p>
                                        </div>
                                    )}
                                    
                                    <div className="w-full bg-slate-900 border border-slate-700/50 rounded-xl p-4 flex justify-between items-center mt-6">
                                        <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">AI Threat Score</span>
                                        <span className={`font-mono text-xl font-bold tracking-tight ${result.status === 'BLOCKED' ? 'text-rose-400' : 'text-emerald-400'}`}>
                                            {(result.probability * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            <style dangerouslySetInnerHTML={{__html: `
                @keyframes scan {
                    0% { left: -50%; }
                    100% { left: 100%; }
                }
            `}} />
        </div>
    );
}
