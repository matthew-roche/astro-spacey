import { useRef, useEffect } from "react";


export function Modal({ open, onClose, title, content }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    if (open) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      aria-modal="true"
      role="dialog"
      aria-labelledby="modal-title"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="absolute inset-0 bg-black/40" />
      <div
        ref={dialogRef}
        className="relative z-10 w-full max-w-3xl rounded-2xl bg-white p-6 shadow-xl"
      >
        <div className="flex items-start justify-between gap-4">
          <h2 id="modal-title" className="text-lg font-semibold text-gray-900">
            {title}
          </h2>
          <button
            onClick={onClose}
            className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1 text-sm text-gray-600 hover:bg-gray-50 cursor-pointer"
          >
            <i class="ri-close-large-line"></i>
          </button>
        </div>
        <div className="p-5">
          <div className="max-h-[70vh] overflow-y-auto pr-1">
            <div className="text-[15px] leading-7 text-gray-700">
              {content}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
