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
    static_assert(indexible<NestedObject>);
    static_assert(indexible<LibraryObject>);
    static_assert(not std::is_rvalue_reference_v<NestedObject>);
#endif

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr AdapterBase() requires std::default_initializable<NestedObject>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<NestedObject>, int> = 0>
    constexpr AdapterBase()
#endif
      : m_nested_object {} {}


    /**
     * \brief Construct from a compatible indexible type.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
    template<indexible Arg> requires (not std::is_base_of_v<Derived, std::decay_t<Arg>>) and
      std::constructible_from<NestedObject, Arg&&>
    constexpr explicit AdapterBase(Arg&& arg) : m_nested_object {std::forward<Arg>(arg)} {}
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not std::is_base_of_v<Derived, std::decay_t<Arg>>), int> = 0>
    constexpr explicit AdapterBase(Arg&& arg, typename std::enable_if<std::is_constructible<NestedObject, Arg&&>::value>::type* = 0)
      : m_nested_object {std::forward<Arg>(arg)} {}
#endif

  protected:

    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not std::is_base_of_v<Derived, std::decay_t<Arg>>) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<Derived, std::decay_t<Arg>>) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&>, int> = 0>
#endif
    constexpr AdapterBase& operator=(Arg&& arg)
    {
      m_nested_object = std::forward<Arg>(arg);
      return *this;
    }

  public:

    /**
     * \brief Get the nested object.
     */
#ifdef __cpp_explicit_this_parameter 
    template<typename Self>
    constexpr NestedObject& nested_object(this Self&& self) { return std::forward<Self>(self).m_nested_object; }
#else
    constexpr NestedObject& nested_object() & { return m_nested_object; }

    /// \overload
    constexpr const NestedObject& nested_object() const & { return m_nested_object; }

    /// \overload
    constexpr NestedObject&& nested_object() && { return std::move(*this).m_nested_object; }

    /// \overload
    constexpr const NestedObject&& nested_object() const && { return std::move(*this).m_nested_object; }
#endif


    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived, Indices></code>, the component can be directly assigned.
     */
#ifdef __cpp_explicit_this_parameter 
    template<typename Self, typename Indices> requires
      requires(Self&& self, const Indices& indices) {
        {get_component(std::forward<Self>(self), indices)} -> values::scalar; }
    constexpr values::scalar auto operator[](this Self&& self, const Indices& indices)
    {
      if constexpr (writable_by_component<Self, Indices>) return ElementAccessor(std::forward<Self>(self), indices);
      else return get_component(std::forward<Self>(self), indices);
    }
#else
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(Derived& derived, const Indices& indices) {{get_component(derived, indices)} -> values::scalar; }
#else
    template<typename Indices, std::enable_if_t<
      values::scalar<decltype(get_component(std::declval<Derived&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator[](const Indices& indices) &
    {
      if constexpr (writable_by_component<Derived, Indices>) return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }

    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(Derived&& derived, const Indices& indices) {{get_component(derived, indices)} -> values::scalar; }
#else
    template<typename Indices, std::enable_if_t<
      values::scalar<decltype(get_component(std::declval<Derived&&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator[](const Indices& indices) &&
    {
      if constexpr (writable_by_component<Derived&&>) return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }

    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(const Derived& derived, const Indices& indices) {{get_component(derived, indices)} -> values::scalar; }
#else
    template<typename Indices, std::enable_if_t<
      values::scalar<decltype(get_component(std::declval<const Derived&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator[](const Indices& indices) const &
    {
      return get_component(static_cast<const Derived&>(*this), indices);
    }

    /// \overload
#ifdef __cpp_lib_concepts
    template<typename Indices> requires
      requires(const Derived&& derived, const Indices& indices) {{get_component(derived, indices)} -> values::scalar; }
#else
    template<typename Indices, std::enable_if_t<
      values::scalar<decltype(get_component(std::declval<const Derived&&>(), std::declval<const Indices&>()))>, int> = 0>
#endif
    constexpr auto operator[](const Indices& indices) const &&
    {
      return get_component(static_cast<const Derived&&>(*this), indices);
    }
#endif


#if defined(__cpp_explicit_this_parameter) and defined(__cpp_multidimensional_subscript)
    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived></code>, the component can be directly assigned.
     */
    template<typename Self, values::index...I> requires
      requires(Self&& self, const std::array<std::size_t, sizeof...(I)>& indices) { 
        {get_component(std::forward<Self>(self), indices)} -> values::scalar; }
    constexpr values::scalar auto operator[](this Self&& self, I&&...i)
    {
      auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      if constexpr (writable_by_component<Self, std::array<std::size_t, sizeof...(I)>>)
        return ElementAccessor(std::forward<Self>(self), indices);
      else 
        return get_component(std::forward<Self>(self), indices);
    }
#endif

  private:

    NestedObject m_nested_object; //< The nested matrix.

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP
