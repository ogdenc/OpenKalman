/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of AdapterBase.
 */

#ifndef OPENKALMAN_ADAPTERBASE_HPP
#define OPENKALMAN_ADAPTERBASE_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Base class for adapters.
   * \tparam Derived The fully derived adapter.
   * \tparam NestedObject The nested object, which can be const or an lvalue reference.
   * \tparam LibraryObject Any object from the library to which this adapter is to be associated.
   */
#ifdef __cpp_concepts
  template<typename Derived, indexible NestedObject, indexible LibraryObject = NestedObject> requires
    (not std::is_rvalue_reference_v<NestedObject>)
#else
  template<typename Derived, typename NestedObject, typename LibraryObject = NestedObject>
#endif
  struct AdapterBase : internal::library_base_t<Derived, LibraryObject>
  {

#ifndef __cpp_concepts
    static_assert(not std::is_rvalue_reference_v<NestedObject>);
#endif


    /**
     * \internal
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr AdapterBase() noexcept requires self_contained<NestedObject> and std::default_initializable<NestedObject>
#else
    template<typename T = NestedObject, std::enable_if_t<self_contained<T> and
      std::is_default_constructible<NestedObject>::value, int> = 0>
    constexpr AdapterBase() noexcept
#endif
      : m_arg {} {}


    /**
     * \internal
     * \brief Construct from a compatible indexible type.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Arg> requires (not std::is_base_of_v<AdapterBase, std::decay_t<Arg>>) and
      std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<AdapterBase, std::decay_t<Arg>> and
      std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    constexpr explicit AdapterBase(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \brief Get the nested object.
     */
    constexpr NestedObject&  nested_object() & noexcept { return m_arg; }


    /// \overload
    constexpr const NestedObject&  nested_object() const & noexcept { return m_arg; }


    /// \overload
    constexpr NestedObject&& nested_object() && noexcept { return std::move(*this).m_arg; }


    /// \overload
    constexpr const NestedObject&& nested_object() const && noexcept { return std::move(*this).m_arg; }


    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived, Indices></code>, the component can be directly assigned.
     */
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(Derived& derived, const Indices& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename Indices, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) &
    {
      if constexpr (writable_by_component<Derived, Indices>) return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(Derived&& derived, const Indices& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename Indices, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) &&
    {
      if constexpr (writable_by_component<Derived&&>) return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(const Derived& derived, const Indices& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename Indices, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) const &
    {
      return get_component(static_cast<const Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(const Derived&& derived, const Indices& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename Indices, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) const &&
    {
      return get_component(static_cast<const Derived&&>(*this), indices);
    }


    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived></code>, the component can be directly assigned.
     */
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(Derived& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator()(I&&...i) &
    {
      auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, sizeof...(I)>>)
        return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(Derived&& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator()(I&&...i) &&
    {
      auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, sizeof...(I)>>)
        return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(const Derived& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator()(I&&...i) const &
    {
      auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      return get_component(static_cast<const Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(const Derived&& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator()(I&&...i) const &&
    {
      auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      return get_component(static_cast<const Derived&&>(*this), indices);
    }


    /**
     * \brief Access the component at a set of indices
     * \return If <code>writable_by_component<Derived></code>, the component can be directly assigned.
     */
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(Derived& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator[](I&&...i) &
    {
      return operator()(std::forward<I>(i)...);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(Derived&& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<Derived&&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator[](I&&...i) &&
    {
      return operator()(std::forward<I>(i)...);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(const Derived& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator[](I&&...i) const &
    {
      return operator()(std::forward<I>(i)...);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      requires(const Derived&& derived, std::array<std::size_t, sizeof...(I)>& indices) {{get_component(derived, indices)} -> scalar_constant; }
#else
    template<typename...I, std::enable_if_t<
      scalar_constant<decltype(get_component(std::declval<const Derived&&>(), std::declval<std::array<std::size_t, sizeof...(I)>&>()))>, int> = 0>
#endif
    constexpr auto operator[](I&&...i) const &&
    {
      return operator()(std::forward<I>(i)...);
    }

  private:

    NestedObject m_arg; //< The nested matrix.

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP
