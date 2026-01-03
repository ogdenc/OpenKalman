/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "linear-algebra/traits/internal/library_base.hpp"
#include "linear-algebra/traits/access.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Base class for adapters.
   * \tparam Derived The fully derived adapter.
   * \tparam Nested The nested object, which can be const or an lvalue reference.
   * \tparam LibraryObject Any object from the library to which this adapter is to be associated.
   */
#ifdef __cpp_concepts
  template<typename Derived, indexible Nested, indexible LibraryObject = Nested> requires
    (not std::is_rvalue_reference_v<Nested>)
#else
  template<typename Derived, typename Nested, typename LibraryObject = Nested>
#endif
  struct AdapterBase : library_base_t<Derived, LibraryObject>
  {

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
    static_assert(indexible<LibraryObject>);
    static_assert(not std::is_rvalue_reference_v<Nested>);
#endif

    /**
     * \brief Default constructor.
     */
    constexpr
    AdapterBase() = default;


    /**
     * \brief Construct from the nested type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires
      (not std::is_base_of_v<Derived, std::decay_t<Arg>>) and
      std::constructible_from<Nested, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      (not std::is_base_of_v<Derived, std::decay_t<Arg>>) and
      stdex::constructible_from<Nested, Arg&&>, int> = 0>
#endif
    constexpr explicit
    AdapterBase(Arg&& arg) : nested_ {std::forward<Arg>(arg)} {}


    /**
     * \brief Get the nested object.
     */
#ifdef __cpp_explicit_this_parameter 
    template<typename Self>
    constexpr decltype(auto) nested_object(this Self&& self) { return std::forward<Self>(self).nested_; }
#else
    constexpr Nested& nested_object() & { return nested_; }

    /// \overload
    constexpr const Nested& nested_object() const & { return nested_; }

    /// \overload
    constexpr Nested&& nested_object() && { return std::move(*this).nested_; }

    /// \overload
    constexpr const Nested&& nested_object() const && { return std::move(*this).nested_; }
#endif


    /**
     * \brief Access a component using a collection or pack of indices.
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, typename...I> requires
      requires(Self&& s, I&&...i) { access(std::forward<Self>(s), std::forward<I>(i)...); }
    constexpr values::value decltype(auto)
    operator[](this Self&& self, I&&...i)
    {
      return access(std::forward<Self>(self), std::forward<I>(i)...);
    }
#else
    template<typename...I, std::enable_if_t<
      values::value<decltype(access(std::declval<Derived&>(), std::declval<I>()...))>, int> = 0>
    constexpr decltype(auto) operator[](I&&...i) &
    {
      return access(static_cast<Derived&>(*this), std::forward<I>(i)...);
    }

    /// \overload
    template<typename...I, std::enable_if_t<
      values::value<decltype(access(std::declval<const Derived&>(), std::declval<I>()...))>, int> = 0>
    constexpr decltype(auto) operator[](I&&...i) const &
    {
      return access(static_cast<const Derived&>(*this), std::forward<I>(i)...);
    }

    /// \overload
    template<typename...I, std::enable_if_t<
      values::value<decltype(access(std::declval<Derived&&>(), std::declval<I>()...))>, int> = 0>
    constexpr decltype(auto) operator[](I&&...i) &&
    {
      return access(static_cast<Derived&&>(*this), std::forward<I>(i)...);
    }

    /// \overload
    template<typename...I, std::enable_if_t<
      values::value<decltype(access(std::declval<const Derived&&>(), std::declval<I>()...))>, int> = 0>
    constexpr decltype(auto) operator[](I&&...i) const &&
    {
      return access(static_cast<const Derived&&>(*this), std::forward<I>(i)...);
    }
#endif

  private:

    Nested nested_;

  };

}

#endif
