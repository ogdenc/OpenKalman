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

    /**
     * \brief Construct from a non-library object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires std::constructible_from<Base, Arg&&> and
      (not std::is_base_of_v<LibraryWrapper, std::decay_t<Arg>>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and std::is_constructible_v<Base, Arg&&> and
      (not std::is_base_of_v<LibraryWrapper, std::decay_t<Arg>>), int> = 0>
#endif
    explicit LibraryWrapper(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /**
     * \brief Move constructor.
     */
    LibraryWrapper(LibraryWrapper&& arg) = default;


    /**
     * \brief Copy constructor.
     */
    LibraryWrapper(const LibraryWrapper& arg) = default;


    /**
     * \brief Move assignment operator.
     */
    LibraryWrapper& operator=(LibraryWrapper&& arg)
    {
      if (this != &arg) this->nested_object() = std::move(arg).nested_object();
      return *this;
    }


    /**
     * \brief Copy assignment operator.
     */
    LibraryWrapper& operator=(const LibraryWrapper& arg)
    {
      if (this != &arg) this->nested_object() = arg.nested_object();
      return *this;
    }


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not std::is_base_of_v<LibraryWrapper, std::decay_t<Arg>>) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<LibraryWrapper, std::decay_t<Arg>>) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>, int> = 0>
#endif
    LibraryWrapper& operator=(Arg&& arg) noexcept
    {
      this->nested_object() = to_native_matrix<NestedObject>(std::forward<Arg>(arg));
      return *this;
    }

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARYWRAPPER_HPP