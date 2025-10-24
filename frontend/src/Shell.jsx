import { NavLink, Routes, Route, Link } from 'react-router-dom';
import Home from "./feature/Home"
import ChangeLog from './feature/ChangeLog';
import { useEffect, useState } from "react";

export default function Shell() {
    const navBtnDefaultStyling = "py-2 px-4 rounded-full text-[#111827] hover:underline underline-offset-4 decoration-accent transition-colors duration-150 font-space-grotesk font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 hover:ring-1 hover:ring-accent"
    const navBtnActiveStyling = "py-2 px-4 rounded-full text-[#111827] font-space-grotesk font-medium underline decoration-accent decoration-2 underline-offset-4 hover:ring-1 hover:ring-accent"
    const [healthy, SetHealthy] = useState(false)

    useEffect(()=>{
        fetch(`/api/health`, {
            headers: {
                "Accept": "application/json"
            }
        })
        .then(response => response.json())
        .then(result => SetHealthy(result?.status == "healthy"))
        .catch(console.error)
    }, [])
    
    return (
        <div className='min-h-screen flex flex-col'>
            <div className="flex-none grid grid-row">
                <div className="col-span-3 justify-items-center-safe">
                    <p className="text-xl md:text-2xl font-space-grotesk text-center pt-2 md:pt-6 tracking-wider">SpaceY<i className={healthy ? "ri-rocket-2-fill text-blue-800": "ri-rocket-2-fill text-black"}></i>
                    </p>
                    <div className="absolute top-2 right-4 flex items-center gap-2">
                        <span className="text-[9px] sm:text-xs">Base Station Status</span>
                        <i className={healthy ? "ri-circle-fill text-green-600 animate-pulse duration-900" : "ri-circle-fill text-red-700"}></i>
                        
                    </div>
                </div>
                <nav className={"col-span-3 flex py-2 px-3 gap-1 bg-white border-b-2 border-accent"}>
                    <NavLink to="/" className={({ isActive }) => { return isActive ? navBtnActiveStyling : navBtnDefaultStyling }}>Explore</NavLink >
                    <NavLink to="/changes" className={({ isActive }) => { return isActive ? navBtnActiveStyling : navBtnDefaultStyling }}>ChangeLog</NavLink>
                </nav>
            </div>
            <div className='grow'>
                <Routes>
                    <Route path="/" element={<Home/>} />
                    <Route path="/changes" element={<ChangeLog />} />
                </Routes>
            </div>
            <footer className={'flex-none h-20px ps-4 pb-2 mt-1 sm:pb-4 pt-4 text-[#111827] border-t-2 border-accent'}>
                <div className='grid grid-cols-2'>
                    <div className='flex flex-wrap'>
                        <span className='sm:ps-3'>2025 AIT500 G4 - Project Demo (version: alpha-0.0.3)</span>
                    </div>
                    <div className='flex flex-row justify-end pe-3'>
                        <span className="pe-4">Contact us:&nbsp;<a href="mailto:xyz@spacey.com" className="hover:text-accent-hover underline-offset-2">xyz@spacey.com</a></span>
                        <span className='pe-1 cursor-pointer content-center ' onClick={() => redirectToLinkedIn()}>
                            <svg class="w-7 h-7" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M18.3362 18.339H15.6707V14.1622C15.6707 13.1662 15.6505 11.8845 14.2817 11.8845C12.892 11.8845 12.6797 12.9683 12.6797 14.0887V18.339H10.0142V9.75H12.5747V10.9207H12.6092C12.967 10.2457 13.837 9.53325 15.1367 9.53325C17.8375 9.53325 18.337 11.3108 18.337 13.6245V18.339H18.3362ZM7.00373 8.57475C6.14573 8.57475 5.45648 7.88025 5.45648 7.026C5.45648 6.1725 6.14648 5.47875 7.00373 5.47875C7.85873 5.47875 8.55173 6.1725 8.55173 7.026C8.55173 7.88025 7.85798 8.57475 7.00373 8.57475ZM8.34023 18.339H5.66723V9.75H8.34023V18.339ZM19.6697 3H4.32923C3.59498 3 3.00098 3.5805 3.00098 4.29675V19.7033C3.00098 20.4202 3.59498 21 4.32923 21H19.6675C20.401 21 21.001 20.4202 21.001 19.7033V4.29675C21.001 3.5805 20.401 3 19.6675 3H19.6697Z"></path></svg>
                        </span>
                        <span>SpaceY</span>
                    </div>
                </div>
            </footer>
        </div>
    )
}