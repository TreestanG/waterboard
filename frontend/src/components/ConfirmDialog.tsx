import Modal from "./Modal";

type Props = {
  title: string;
  body: string;
  confirmLabel: string;
  onCancel: () => void;
  onConfirm: () => void;
};

export default function ConfirmDialog({ title, body, confirmLabel, onCancel, onConfirm }: Props) {
  return (
    <Modal title={title} onClose={onCancel} size="sm">
      <p className="muted">{body}</p>
      <div className="modal-actions">
        <button className="btn" onClick={onCancel}>
          Cancel
        </button>
        <button className="btn btn-danger-solid" onClick={onConfirm} autoFocus>
          {confirmLabel}
        </button>
      </div>
    </Modal>
  );
}
