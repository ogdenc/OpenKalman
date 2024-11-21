/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for internal::LibraryWrapper
 */

#ifndef OPENKALMAN_LIBRARYWRAPPER_HPP
#define OPENKALMAN_LIBRARYWRAPPER_HPP


namespace OpenKalman::internal
{

#ifdef __cpp_concepts
  template<indexible NestedObject, indexible LibraryObject>
#else
  template<typename NestedObject, typename LibraryObject>
#endif
  struct LibraryWrapper : AdapterBase<LibraryWrapper<NestedObject, LibraryObject>, NestedObject, LibraryObject>
  {
  private:

    using Base = AdapterBase<LibraryWrapper, NestedObject, LibraryObject>;

  public:

    using Base::Base;


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires
      (std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&> or
        std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>)
#else
    template<typename Arg, std::enable_if_t<
      (std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&> or
        std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>), int> = 0>
#endif
    constexpr LibraryWrapper& operator=(Arg&& arg)
    {
#ifdef __cpp_concepts
      if constexpr (std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&>)
#else
      if constexpr (std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&>)
#endif
        Base::operator=(std::forward<Arg>(arg));
      else
        Base::operator=(to_native_matrix<NestedObject>(std::forward<Arg>(arg)));
      return *this;
    }


    /**
     * \brief Convert to the nested object.
     */
    constexpr operator NestedObject& () & { return this->nested_object(); }

    /// \overload
    constexpr operator const NestedObject& () const & { return this->nested_object(); }

    /// \overload
    constexpr operator NestedObject&& () && { return std::move(*this).nested_object(); }

    /// \overload
    constexpr operator const NestedObject&& () const && { return std::move(*this).nested_object(); }

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARYWRAPPER_HPP