/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
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

#include<iostream>

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedObject, typename...InternalizedParameters>
#else
  template<typename NestedObject, typename...InternalizedParameters>
#endif
  struct SelfContainedWrapper : internal::library_base_t<SelfContainedWrapper<NestedObject, InternalizedParameters...>, NestedObject>
  {
  private:

    using Base = internal::library_base_t<SelfContainedWrapper, NestedObject>;

  public:

    /**
     * \overload
     * \brief Construct from a set of parameters, which may be stored in this object to extend their lifetime.
     * \tparam Ps Parameters from which to construct the nested object.
     * Each Ps corresponds to an associated InternalizedParameters type.
     */
#ifdef __cpp_concepts
    template<typename...Ps> requires (sizeof...(Ps) != 1 or (... and (not std::same_as<std::decay_t<Ps>, SelfContainedWrapper>)))
#else
    template<typename...Ps, std::enable_if_t<
      (sizeof...(Ps) != 1 or (... and (not std::is_same_v<std::decay_t<Ps>, SelfContainedWrapper>))), int> = 0>
#endif
    explicit SelfContainedWrapper(Ps&&...ps) :
      internalized_parameters {std::forward<Ps>(ps)...},
      wrapped_expression {std::apply([](auto&&...p){ return NestedObject {p...}; }, internalized_parameters)} {}


    /**
     * \brief Move constructor.
     */
    SelfContainedWrapper(SelfContainedWrapper&& arg) noexcept
      : internalized_parameters {std::move(arg).internalized_parameters},
        wrapped_expression {std::apply([](auto&&...p){ return NestedObject {p...}; }, internalized_parameters)} {}


    /**
     * \brief Copy constructor.
     */
    SelfContainedWrapper(const SelfContainedWrapper& arg)
      : internalized_parameters {arg.internalized_parameters},
        wrapped_expression {std::apply([](auto&&...p){ return NestedObject {p...}; }, internalized_parameters)} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      wrapped_expression = to_native_matrix<NestedObject>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Move assignment operator.
     */
    SelfContainedWrapper& operator=(SelfContainedWrapper&& arg) = default;


    /**
     * \brief Copy assignment operator.
     */
    SelfContainedWrapper& operator=(const SelfContainedWrapper& arg)
    {
      internalized_parameters = arg.internalized_parameters;
      return *this;
    }


    /**
     * \brief Get the nested object.
     */
    auto& nested_object() & noexcept { return wrapped_expression; }

    /// \overload
    const auto& nested_object() const & noexcept { return wrapped_expression; }

    /// \overload
    auto&& nested_object() && noexcept { return std::move(*this).wrapped_expression; }

    /// \overload
    const auto&& nested_object() const && noexcept { return std::move(*this).wrapped_expression; }


    /**
     * \brief Convert to the nested object
     */
    operator NestedObject() & noexcept { return wrapped_expression; }

    /// \overload
    operator NestedObject() const & noexcept { return wrapped_expression; }

    /// \overload
    operator NestedObject() && noexcept { return std::move(*this).wrapped_expression; }

    /// \overload
    operator NestedObject() const && noexcept { return std::move(*this).wrapped_expression; }

  private:

    std::tuple<InternalizedParameters...> internalized_parameters;

    NestedObject wrapped_expression;

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SELFCONTAINEDWRAPPER_HPP