import { useState, useEffect } from "react";
import { loadIndex, rowsByIds } from "./loadarrow";
import { search } from './searchreq'
import { Modal } from './Modal';
 
export default function Home() {
    const [index, setIndex] = useState(null); //spacenews 
    const [filteredRecords, setFilteredRecords] = useState([]) // backend filtered
    const [query, setQuery] = useState(""); //search query
    const [showAI, setShowAI] = useState(false); //show ai response if ai answered
    const [aiMode, setAIMode] = useState(false) //use ai in search
    const [aiResult, setAiResult] = useState("AI response here")

    const [activeItem, setActiveItem] = useState(null) //list on click
    const [openModal, setModalOpen] = useState(false) //dialog

    const [loading, setLoading] = useState(false) //backend response

    const API_URL = import.meta.env.VITE_API_URL
    const API_SEC = import.meta.env.VITE_API_SEC

    useEffect(() => {
        setLoading(true)
        loadIndex(`/api/data`, API_SEC).then(setIndex).catch(console.error).finally(() => setLoading(false));
    }, [API_URL]);

    useEffect(()=> {
        console.log(filteredRecords)
    }, [filteredRecords])

    function handleSearch() {
        setLoading(true)
        search(`/api`, query, aiMode, API_SEC).then((result) => {
            if (result.ai_answered == true){
                setShowAI(true)
                setAiResult(result.ai_text)
            }else{
                setShowAI(false)
            }

            // local filtering by backend ids
            let rows = rowsByIds(index, result.response)
            setFilteredRecords(rows)
        }).catch(console.error).finally(() => setLoading(false))
    }

    function truncate(str, n = 180) {
        if (!str) return "";
        return str.length > n ? str.slice(0, n - 1).trimEnd() + "…" : str;
    }

    return <div className='justify-items-center-safe'>
        <div className='w-full max-w-7xl'>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2 ps-1 pt-1">
                {/* Left side: title and search bar */}
                <div>
                    <h1 className="text-2xl font-semibold mb-4">Explore with us.</h1>
                    <div className="relative flex items-center">
                        <input
                            type="text"
                            placeholder="Type something to search"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => {if (e.key === 'Enter') handleSearch()}}
                            className="w-full rounded-full border border-gray-200 px-5 py-3 pr-14 text-[15px] outline-none shadow-sm focus:border-gray-300 focus:ring-4 focus:ring-black/5"
                        />
                        <button
                            onClick={() => handleSearch()}
                            className="absolute right-1 inline-flex h-10 w-10 items-center justify-center rounded-full bg-gray-900 text-white shadow transition active:scale-95 hover:bg-gray-800"
                        >
                            <i class="ri-search-line"></i>
                        </button>
                    </div>
                </div>

                {/* Right side: placeholder for future content (AI toggle or results) */}
                <div className="flex items-center gap-3 h-full lg:pt-11">
                    <input type="checkbox" id="ai-toggle" className="h-5 w-5 accent-gray-900 align-middle" checked={aiMode} onChange={() => setAIMode(!aiMode)} />
                    <label htmlFor="ai-toggle" className="cursor-pointer">
                        <span className="block text-sm font-medium text-gray-900">Experimental AI</span>
                        <span className="block text-xs text-gray-500">Try our new AI search</span>
                    </label>

                    {loading && (
                        <i className="ri-refresh-line animate-spin text-gray-500 text-3xl ml-auto" />
                    )}
                </div>
            </div>

            {showAI && (
            <div className="px-1">
                <div className="mt-6 rounded-lg border border-emerald-600/70 bg-emerald-50 p-5 text-emerald-900">
                    {/* <div className="font-semibold mb-1">AI Result:</div> */}
                    <div className="text-sm">AI Answer:&nbsp;{aiResult}</div>
                </div>
            </div>)}
            <ul className="mt-6 space-y-6">
                {filteredRecords?.length ? (
                    filteredRecords.map((item, idx) => (
                    <li
                    key={item.id ?? idx}
                    className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm hover:shadow transition-shadow cursor-pointer"
                    onClick={()=>{setActiveItem(item); setModalOpen(true);}}
                    >
                        <h3 className="text-lg font-semibold text-gray-900">{item.title}</h3>
                        <p className="mt-2 text-sm leading-6 text-gray-600">{truncate(item.content)}</p>
                    </li>
                    ))
                    ) : (
                    // Empty state / placeholder
                    <li className="rounded-2xl border border-dashed border-gray-300 p-6 text-sm text-gray-500">
                    No results yet. Run a search to populate the list.
                    </li>
                )}
            </ul>
        </div>
         <Modal open={openModal} onClose={() => setModalOpen(false)} title={activeItem?.title} content={activeItem?.content}></Modal>
    </div>
}