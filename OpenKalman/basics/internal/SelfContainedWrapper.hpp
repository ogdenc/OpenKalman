/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for \ref internal::SelfContainedWrapper "SelfContainedWrapper"
 */

#ifndef OPENKALMAN_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_SELFCONTAINEDWRAPPER_HPP

#include<memory>

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedObject, typename...Parameters>
#else
  template<typename NestedObject, typename...Parameters>
#endif
  struct SelfContainedWrapper : AdapterBase<SelfContainedWrapper<NestedObject, Parameters...>, NestedObject>
  {
  private:

    template<typename T>
    struct is_SelfContainedWrapper : std::false_type {};

    template<typename N, typename...Ps>
    struct is_SelfContainedWrapper<SelfContainedWrapper<N, Ps...>> : std::true_type {};

    static_assert(not is_SelfContainedWrapper<NestedObject>::value);
    static_assert(not std::is_reference_v<NestedObject>);
    static_assert((... and (not std::is_reference_v<Parameters>)));

    using Base = AdapterBase<SelfContainedWrapper, NestedObject>;

  public:

    /**
     * \brief Construct from a set of arguments, some of which may be references or pointers to managed parameters.
     * \param parameters Pointers to any the parameters to be internalized. The wrapper takes ownership of the pointer
     * and manages its storage. The pointer will be deleted when this wrapper is deleted.
     * \tparam Arg An instance of NestedObject, which may include references to the parameters.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires
      std::constructible_from<Base, Arg&&> and (not std::is_base_of_v<SelfContainedWrapper, std::decay_t<Arg>>)
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<Base, Arg&&> and
      (not std::is_base_of_v<SelfContainedWrapper, std::decay_t<Arg>>), int> = 0>
#endif
    explicit SelfContainedWrapper(Arg&& arg, Parameters*...parameters)
      : Base {std::forward<Arg>(arg)}, internalized_parameters {parameters...} {}


    /**
     * \brief Convert to the nested object
     */
    operator NestedObject() & noexcept { return this->nested_object(); }

    /// \overload
    operator NestedObject() const & noexcept { return this->nested_object(); }

    /// \overload
    operator NestedObject() && noexcept { return std::move(*this).nested_object(); }

    /// \overload
    operator NestedObject() const && noexcept { return std::move(*this).nested_object(); }

  private:

    std::tuple<std::shared_ptr<Parameters>...> internalized_parameters;

  };


  // ------------------ //
  //  Deduction Guide  //
  // ------------------ //

#ifdef __cpp_concepts
  template<indexible Arg, typename...Ps>
#else
  template<typename Arg, typename...Ps>
#endif
  SelfContainedWrapper(Arg&&, Ps*...) -> SelfContainedWrapper<std::remove_reference_t<Arg>, std::remove_reference_t<Ps>...>;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SELFCONTAINEDWRAPPER_HPP