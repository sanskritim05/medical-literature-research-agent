interface Props {
  checked: boolean;
  onCheckedChange: (value: boolean) => void;
}

export function Switch({ checked, onCheckedChange }: Props) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onCheckedChange(e.target.checked)}
      />
      <span className="switch-slider" />
    </label>
  );
}
